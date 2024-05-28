use crate::{get_config, thread_trait::Processor};
use anyhow::{anyhow, Result};
use intelligent_sight_lib::{
    convert_rgb888_3dtensor, convert_rgb888_3dtensor_with_padding, create_context, create_engine,
    infer, postprocess_classify, postprocess_classify_destroy, set_input, set_output,
    transfer_host_to_device, transfer_host_to_host, Detection, DetectionBuffer, ImageBuffer,
    Reader, Searcher, TensorBuffer, UnifiedTrait, Writer,
};
use log::{debug, error, log_enabled};
use opencv::{self as cv, core::*};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
};

pub struct ClassificationThread {
    input_buffer: Reader<DetectionBuffer>,
    output_buffer: Writer<DetectionBuffer>,
    query_buffer: Searcher<ImageBuffer>,
    stop_sig: Arc<AtomicBool>,
}

impl Processor for ClassificationThread {
    type Output = DetectionBuffer;

    fn get_output_buffer(&self) -> Reader<Self::Output> {
        self.output_buffer.get_reader()
    }

    fn start_processor(self) -> std::thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut img_mat = Mat::default();
            let mut img = ImageBuffer::new(320, 320).unwrap();
            let mut infer_input = TensorBuffer::new(vec![3, 320, 320]).unwrap();
            let mut infer_output = TensorBuffer::new(vec![1, 16, 2100]).unwrap();
            set_input(1, &mut infer_input).unwrap();
            set_output(1, &mut infer_output).unwrap();
            let mut cnt = 0;
            let mut start = std::time::Instant::now();
            while self.stop_sig.load(Ordering::Relaxed) == false {
                let Some(lock_input) = self.input_buffer.read() else {
                    if self.stop_sig.load(Ordering::Relaxed) == false {
                        error!("ClassificationThread: Failed to get input");
                    }
                    break;
                };
                let Some(mut image) = self.query_buffer.search(lock_input.timestamp) else {
                    debug!("missing image");
                    continue;
                };
                let mat = unsafe {
                    match Mat::new_rows_cols_with_data_unsafe(
                        image.height as i32,
                        image.width as i32,
                        CV_8UC3,
                        image.host() as *mut std::ffi::c_void,
                        image.width as usize * 3 * std::mem::size_of::<u8>(),
                    ) {
                        Ok(mat) => mat,
                        Err(err) => {
                            error!(
                                "ClassificationThread: Failed to create Mat, detail: {}",
                                err
                            );
                            break;
                        }
                    }
                };

                let mut lock_output = self.output_buffer.write();
                let mut iter = lock_output.iter_mut();
                for Detection {
                    x, y, w, h, conf, ..
                } in lock_input.iter().take(lock_input.size)
                {
                    let rect = Rect_::new(
                        (x - w / 2.0).round() as i32,
                        (y - h / 2.0).round() as i32,
                        w.round() as i32,
                        h.round() as i32,
                    );
                    let slice = match mat.roi(rect) {
                        Ok(slice) => slice,
                        Err(err) => {
                            error!("ClassificationThread: Failed to get ROI, detail: {}", err);
                            break;
                        }
                    };
                    let size = if rect.width > rect.height {
                        rect.width
                    } else {
                        rect.height
                    } as f64;
                    let factor = 320.0 / size;

                    let size = match img_mat.size() {
                        Ok(size) => size,
                        Err(err) => {
                            error!("ClassificationThread: Failed to get size, detail: {}", err);
                            self.stop_sig.store(true, Ordering::Relaxed);
                            return;
                        }
                    };

                    let width = (size.width as f64 * factor) as i32;
                    let height = (size.height as f64 * factor) as i32;

                    if let Err(err) = cv::imgproc::resize(
                        &slice,
                        &mut img_mat,
                        Size_ { width, height },
                        0.0,
                        0.0,
                        cv::imgproc::INTER_LINEAR,
                    ) {
                        error!("ClassificationThread: Resize error occur, detail: {}", err);
                        self.stop_sig.store(true, Ordering::Relaxed);
                        return;
                    }

                    let size = match img_mat.size() {
                        Ok(size) => size,
                        Err(err) => {
                            error!("ClassificationThread: Failed to get size, detail: {}", err);
                            self.stop_sig.store(true, Ordering::Relaxed);
                            return;
                        }
                    };

                    if let Err(err) = transfer_host_to_device(
                        img_mat.data(),
                        img.device().unwrap(),
                        (size.width * size.height * 3) as usize,
                    ) {
                        error!("ClassificationThread: Malloc error occur, detail: {}", err);
                        self.stop_sig.store(true, Ordering::Relaxed);
                        return;
                    }
                    img.width = size.width as u32;
                    img.height = size.height as u32;
                    if let Err(err) =
                        convert_rgb888_3dtensor_with_padding(&mut img, &mut infer_input, 320, 320)
                    {
                        error!(
                            "ClassificationThread: Conversion error occur, detail: {}",
                            err
                        );
                        self.stop_sig.store(true, Ordering::Relaxed);
                        return;
                    }
                    if let Err(err) = infer(1) {
                        error!(
                            "ClassificationThread: Inference error occur, detail: {}",
                            err
                        );
                        self.stop_sig.store(true, Ordering::Relaxed);
                        return;
                    }

                    match postprocess_classify(&mut infer_output, 2100) {
                        Ok(cls) => {
                            let detection = iter.next().unwrap();
                            detection.x = *x;
                            detection.y = *y;
                            detection.w = *w;
                            detection.h = *h;
                            detection.conf = *conf;
                            detection.cls = cls;
                        }
                        Err(err) => {
                            error!(
                                "ClassificationThread: Fail postprocess classification, detail: {}",
                                err
                            );
                            return;
                        }
                    }
                }
                if log_enabled!(log::Level::Debug) {
                    cnt += 1;
                    if cnt == 10 {
                        let end = std::time::Instant::now();
                        let elapsed = end.duration_since(start);
                        debug!(
                            "ClassificationThread: fps: {}",
                            10.0 / elapsed.as_secs_f32()
                        );
                        start = end;
                        cnt = 0;
                    }
                }
            }
            if let Err(err) = postprocess_classify_destroy() {
                error!(
                    "ClassificationThread: Postprocess destroy error occur, detail: {}",
                    err
                );
                self.stop_sig.store(true, Ordering::Relaxed);
                return;
            }
            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}

impl ClassificationThread {
    #[allow(unused)]
    const COLORS: [VecN<f64, 4>; 5] = [
        VecN::new(0.0, 0.0, 255.0, 255.0),
        VecN::new(0.0, 255.0, 0.0, 255.0),
        VecN::new(255.0, 0.0, 0.0, 255.0),
        VecN::new(255.0, 255.0, 0.0, 255.0),
        VecN::new(255.0, 0.0, 255.0, 255.0),
    ];

    pub fn new(
        input_buffer: Reader<DetectionBuffer>,
        query_buffer: Searcher<ImageBuffer>,
        stop_sig: Arc<AtomicBool>,
    ) -> Result<Self> {
        create_engine(
            1,
            "./weights/car_classification.trt",
            "images",
            "output0",
            320,
            320,
        )?;
        create_context(1)?;
        let max_detections = match get_config() {
            Some(config) => config.max_detections,
            None => 25,
        };
        Ok(ClassificationThread {
            input_buffer,
            output_buffer: Writer::new(4, || Ok(DetectionBuffer::new(max_detections as usize)))?,
            query_buffer,
            stop_sig,
        })
    }
}
