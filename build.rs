//! tiny-llm（C++/CUDA）执行后端链接脚本。
//!
//! 里程碑 3：当 `TINY_LLM_DIR` 指向 tiny-llm 的 CMake 构建目录
//! （含 `libtiny_llm.a` 与 `_deps/spdlog-build/libspdlog.a`）时，
//! 链接静态库及其传递依赖（spdlog / CUDA runtime / libstdc++）。
//!
//! 未设置 `TINY_LLM_DIR` 时跳过链接（骨架阶段，`tiny-llm` feature 未启用
//! 时该符号不会在 Rust 侧引用，不会导致链接失败）。

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TINY_LLM_DIR");
    println!("cargo:rerun-if-env-changed=TINY_LLM_MODEL");

    let Ok(dir) = env::var("TINY_LLM_DIR") else {
        println!(
            "cargo:warning=paged-serving: 未设置 TINY_LLM_DIR，跳过 tiny-llm 静态库链接。\
             设置它指向 tiny-llm 的 build 目录以启用真实后端。"
        );
        return;
    };

    let build_dir = PathBuf::from(dir);
    let lib_a = build_dir.join("libtiny_llm.a");
    let spdlog_a = build_dir.join("_deps/spdlog-build/libspdlog.a");
    // 监听静态库与 spdlog 库文件变化，触发重新链接
    println!("cargo:rerun-if-changed={}", lib_a.display());
    if spdlog_a.exists() {
        println!("cargo:rerun-if-changed={}", spdlog_a.display());
    }
    if !lib_a.exists() {
        println!(
            "cargo:warning=paged-serving: TINY_LLM_DIR 下未找到 libtiny_llm.a（{}），跳过链接。",
            lib_a.display()
        );
        return;
    }

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=tiny_llm");

    if spdlog_a.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            build_dir.join("_deps/spdlog-build").display()
        );
        println!("cargo:rustc-link-lib=static=spdlog");
    }

    // CUDA runtime + C++ 标准库（静态 C++ 库的传递依赖）
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
}
