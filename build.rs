use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/hetero_cuda_backend.cu");
    println!("cargo:rerun-if-changed=cuda/hetero_cuda_backend_fallback.cpp");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=CUDAHOSTCXX");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc =
        resolve_tool("NVCC", &["/usr/bin/nvcc"], "nvcc").unwrap_or_else(|| "nvcc".to_string());
    let host_cxx = resolve_tool(
        "CUDAHOSTCXX",
        &["/usr/bin/g++-12", "/usr/bin/g++", "/usr/bin/g++-13"],
        "g++",
    )
    .or_else(|| {
        resolve_tool(
            "CXX",
            &["/usr/bin/g++-12", "/usr/bin/g++", "/usr/bin/g++-13"],
            "g++",
        )
    })
    .unwrap_or_else(|| "g++".to_string());

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let lib_path = out_dir.join("libhetero_cuda_backend.a");
    if command_available(&nvcc) {
        let source = Path::new("cuda/hetero_cuda_backend.cu");
        let output = Command::new(&nvcc)
            .arg("--lib")
            .arg("-allow-unsupported-compiler")
            .arg("-std=c++17")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-ccbin")
            .arg(&host_cxx)
            .arg("-o")
            .arg(&lib_path)
            .arg(source)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "failed to invoke nvcc at `{nvcc}` for {}: {error}",
                    source.display()
                )
            });

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "nvcc failed while building {} with host compiler `{host_cxx}`\nstdout:\n{stdout}\nstderr:\n{stderr}",
                source.display()
            );
        }

        if Path::new("/usr/lib/x86_64-linux-gnu").exists() {
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        }
        if Path::new("/lib/x86_64-linux-gnu").exists() {
            println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        let source = Path::new("cuda/hetero_cuda_backend_fallback.cpp");
        let object_path = out_dir.join("hetero_cuda_backend_fallback.o");
        let compile = Command::new(&host_cxx)
            .arg("-std=c++17")
            .arg("-fPIC")
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&object_path)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "failed to invoke host compiler `{host_cxx}` for fallback backend {}: {error}",
                    source.display()
                )
            });
        if !compile.status.success() {
            let stdout = String::from_utf8_lossy(&compile.stdout);
            let stderr = String::from_utf8_lossy(&compile.stderr);
            panic!(
                "host compiler failed while building fallback backend {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                source.display()
            );
        }

        let archive = Command::new("ar")
            .arg("crus")
            .arg(&lib_path)
            .arg(&object_path)
            .output()
            .unwrap_or_else(|error| panic!("failed to archive fallback CUDA backend: {error}"));
        if !archive.status.success() {
            let stdout = String::from_utf8_lossy(&archive.stdout);
            let stderr = String::from_utf8_lossy(&archive.stderr);
            panic!("archiving fallback CUDA backend failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=hetero_cuda_backend");
}

fn resolve_tool(env_key: &str, preferred_paths: &[&str], fallback_name: &str) -> Option<String> {
    env::var(env_key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            preferred_paths
                .iter()
                .find(|path| Path::new(path).exists())
                .map(|path| (*path).to_string())
        })
        .or_else(|| Some(fallback_name.to_string()))
}

fn command_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
