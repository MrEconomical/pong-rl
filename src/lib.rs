mod config;

use pyo3::prelude::*;

#[pyfunction]
pub fn test() {
    println!("test function");
}

#[pymodule]
fn pong_rl(_: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(test, module)?)?;
    Ok(())
}
