use crate::{thread_trait::Processor, CONFIG};
use anyhow::Result;
use intelligent_sight_lib::{
    postprocess, postprocess_destroy, postprocess_init, postprocess_init_default, DetectionBuffer,
    Reader, TensorBuffer, Writer,
};
use log::{debug, error, info, log_enabled};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

#[cfg(feature = "visualize")]
use std::sync::mpsc;

pub struct PostprocessThread {
    input_buffer: Reader<TensorBuffer>,
    output_buffer: Writer<DetectionBuffer>,
    stop_sig: Arc<AtomicBool>,
    #[cfg(feature = "visualize")]
    detection_tx: std::sync::mpsc::Sender<TensorBuffer>,
}

impl Drop for PostprocessThread {
    fn drop(&mut self) {
        if let Err(err) = postprocess_destroy() {
            error!("PostprocessThread: Failed to release resources: {}", err);
        }
    }
}

impl Processor for PostprocessThread {
    type Output = DetectionBuffer;

    fn get_output_buffer(&self) -> Reader<Self::Output> {
        self.output_buffer.get_reader()
    }

    fn start_processor(self) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let max_detections = unsafe {
                if let Some(Some(config)) = std::ptr::addr_of!(CONFIG).as_ref() {
                    config.max_detections as usize
                } else {
                    25
                }
            };
            let mut cnt = 0;
            let mut start = std::time::Instant::now();
            let mut infer_res = TensorBuffer::new(vec![max_detections, 5]).unwrap();
            while self.stop_sig.load(Ordering::Relaxed) == false {
                let mut lock_input = match self.input_buffer.read() {
                    Some(input) => input,
                    None => {
                        if self.stop_sig.load(Ordering::Relaxed) == false {
                            error!("PostprocessThread: Failed to get input");
                        }
                        break;
                    }
                };
                match postprocess(&mut lock_input, &mut infer_res) {
                    Ok(cnt) => {
                        infer_res.timestamp = lock_input.timestamp;

                        #[cfg(feature = "visualize")]
                        {
                            let mut det = infer_res.clone();
                            det.resize(vec![cnt as usize, 5]);
                            if let Err(err) = self.detection_tx.send(det) {
                                if self.stop_sig.load(Ordering::Relaxed) == false {
                                    error!("PostprocessThread: Failed to send detection: {}", err);
                                }
                                break;
                            }
                        }

                        if cnt == 0 {
                            continue;
                        }
                        drop(lock_input);

                        let mut lock_output = self.output_buffer.write();
                        lock_output.timestamp = infer_res.timestamp;
                        let mut iter = infer_res.iter();
                        for i in 0..cnt as usize {
                            lock_output[i].x = *iter.next().unwrap();
                            lock_output[i].y = *iter.next().unwrap();
                            lock_output[i].w = *iter.next().unwrap();
                            lock_output[i].h = *iter.next().unwrap();
                            lock_output[i].conf = *iter.next().unwrap();
                        }
                        lock_output.size = cnt as usize;
                    }
                    Err(err) => {
                        error!("PostprocessThread: Failed to postprocess: {}", err);
                        break;
                    }
                }
                if log_enabled!(log::Level::Debug) {
                    cnt += 1;
                    if cnt == 10 {
                        let end = std::time::Instant::now();
                        let elapsed = end.duration_since(start);
                        debug!("PostprocessThread: fps: {}", 10.0 / elapsed.as_secs_f32());
                        start = end;
                        cnt = 0;
                    }
                }
            }
            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}

impl PostprocessThread {
    pub fn new(
        input_buffer: Reader<TensorBuffer>,
        stop_sig: Arc<AtomicBool>,
        #[cfg(feature = "visualize")] detection_tx: mpsc::Sender<TensorBuffer>,
    ) -> Result<Self> {
        let max_detections = unsafe {
            if let Some(Some(config)) = std::ptr::addr_of!(CONFIG).as_ref() {
                postprocess_init(
                    config.max_detections,
                    config.confidence_threshold,
                    config.iou_threshold,
                    config.feature_map_size,
                )?;
                config.max_detections as usize
            } else {
                postprocess_init_default()?;
                25
            }
        };

        info!(
            "PostprocessThread: output buffer size: {:?}",
            vec![max_detections]
        );

        Ok(Self {
            input_buffer,
            output_buffer: Writer::new(4, || Ok(DetectionBuffer::new(max_detections)))?,
            stop_sig,
            #[cfg(feature = "visualize")]
            detection_tx,
        })
    }
}
