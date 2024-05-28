use criterion::{criterion_group, criterion_main, Criterion};
use intelligent_sight_lib::{
    create_context, create_engine, infer, release_resources, set_input, set_output, TensorBuffer,
};

fn infer_bench(c: &mut Criterion) {
    create_engine(0, "../model.trt", "images", "output0", 640, 480).unwrap();
    create_context(0).unwrap();
    let mut tensor = TensorBuffer::new(vec![640, 480, 3]).unwrap();
    let mut output = TensorBuffer::new(vec![1, 32, 6300]).unwrap();
    set_input(0, &mut tensor).unwrap();
    set_output(0, &mut output).unwrap();
    c.bench_function("inference", |b| {
        b.iter(|| {
            criterion::black_box({
                infer(0).unwrap();
            })
        })
    });
    release_resources(0).unwrap();
}

criterion_group!(benches, infer_bench);
criterion_main!(benches);
