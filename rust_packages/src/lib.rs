use pyo3::prelude::*;

#[pyfunction]
fn calculate_means(_py: Python, grain_area: u32, grain_domain: Vec<Vec<u32>>) -> PyResult<(u32, u32)>
{
    let mut allx: u32 = 0;
    let mut ally: u32 = 0;
    // println!("grain area: {}", grain_area);
    for i in 0..=grain_area as usize
    {
        allx += grain_domain[i][0];
        ally += grain_domain[i][1];
    }
    return Ok((allx / grain_area, ally / grain_area));
}

#[pyfunction]
fn calculate_distances_sum_from_center(_py: Python, grain_domain: Vec<Vec<i64>>, center_of_mass: Vec<i64>) -> PyResult<i64>
{
    let mut distance_sum_power = 0;
    grain_domain.iter().for_each(|point| {
        distance_sum_power += i64::pow(center_of_mass[0] - point[0], 2) + i64::pow(center_of_mass[1] - point[1], 2)
    });
    return Ok(distance_sum_power);
}

/// A Python module implemented in Rust.
#[pymodule]
fn coefficients(m: &Bound<'_, PyModule>) -> PyResult<()>
{
    m.add_function(wrap_pyfunction!(calculate_means, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_distances_sum_from_center, m)?)?;
    Ok(())
}
