use std::path::PathBuf;
fn main() {
    build_training();
}

fn build_training() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
    println!("cargo:rerun-if-env-changed=CXX");
    if std::env::var_os("CARGO_FEATURE_TRAIN").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=src/learning/tensor.cpp");
    let root = PathBuf::from(
        std::env::var_os("LIBTORCH")
            .unwrap_or_else(|| "/usr/local/lib/python3.12/dist-packages/torch".into()),
    );
    assert!(
        root.join("include/ATen/ATen.h").exists(),
        "Set LIBTORCH to a matching LibTorch installation"
    );
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let abi = std::env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".into());
    assert!(
        ["0", "1"].contains(&abi.as_str()),
        "invalid LibTorch C++ ABI"
    );
    let status = std::process::Command::new(compiler)
        .args([
            "-std=c++17",
            "-O2",
            "-g0",
            "-fPIC",
            "-c",
            "src/learning/tensor.cpp",
        ])
        .arg(format!("-D_GLIBCXX_USE_CXX11_ABI={abi}"))
        .arg("-I")
        .arg(root.join("include"))
        .arg("-I")
        .arg(root.join("include/torch/csrc/api/include"))
        .arg("-o")
        .arg(out.join("tensor.o"))
        .status()
        .expect("start C++ compiler");
    assert!(status.success(), "LibTorch bridge compilation failed");
    assert!(
        std::process::Command::new("ar")
            .arg("crs")
            .arg(out.join("libfarm_tensor.a"))
            .arg(out.join("tensor.o"))
            .status()
            .unwrap()
            .success(),
        "archive tensor bridge"
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=farm_tensor");
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    // CUDA kernels register through library constructors; retain them even though
    // the generic ATen entrypoints are resolved from torch_cpu.
    if root.join("lib/libtorch_cuda.so").exists() {
        println!(
            "cargo:rustc-link-arg=-Wl,--no-as-needed,{},--as-needed",
            root.join("lib/libtorch_cuda.so").display()
        );
    }
    for lib in ["torch", "torch_cpu", "c10", "stdc++"] {
        println!("cargo:rustc-link-lib={lib}");
    }
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        root.join("lib").display()
    );
}
