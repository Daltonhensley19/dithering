#![feature(int_roundings)]

use color::XyzD65;
use serde_json::from_str;
use std::{f32::consts::PI, io::Read, ops::Sub};

use image::{EncodableLayout, ImageBuffer, ImageReader, Luma, Pixel, Rgb};

#[derive(Debug, Clone)]
struct ColorPalette {
    palette: Vec<Rgb<f32>>,
}

impl ColorPalette {
    fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        let mut file = std::fs::File::open(path.as_ref()).expect("Unable to open file");

        let mut file_data = String::with_capacity(file.metadata().unwrap().len() as usize);
        file.read_to_string(&mut file_data)
            .expect("Unable to read file");

        let color_palette: Vec<[f32; 3]> = from_str(&file_data).unwrap();

        let color_palette: Vec<Rgb<f32>> =
            color_palette.into_iter().map(|color| Rgb(color)).collect();

        ColorPalette {
            palette: color_palette,
        }
    }

    fn to_lab(&self) -> ColorPalette {
        let mut color_palette = self.palette.clone();

        for color in color_palette.iter_mut() {
            pixelcolor_to_xyz(color);
            pixelcolor_to_lab(color);
        }

        ColorPalette {
            palette: color_palette,
        }
    }
}

fn pixelcolor_to_lab(pixel: &mut Rgb<f32>) {
    let x_reference = 95.047;
    let y_reference = 100.;
    let z_reference = 108.883;

    fn nonlinear_transform(c: f32) -> f32 {
        let epsion1 = 0.008856;
        let epsion2 = 903.3;
        if c > epsion1 {
            return c.powf(1. / 3.);
        } else {
            return (epsion2 * c + 16.) / 116.;
        }
    }

    let x_norm = pixel.0[0] / x_reference;
    let y_norm = pixel.0[1] / y_reference;
    let z_norm = pixel.0[2] / z_reference;

    let l = 116. * nonlinear_transform(y_norm) - 16.;
    let a = 500. * (nonlinear_transform(x_norm) - nonlinear_transform(y_norm));
    let b = 200. * (nonlinear_transform(y_norm) - nonlinear_transform(z_norm));

    pixel.0 = [l, a, b];
}

fn pixelcolor_to_xyz(pixel: &mut Rgb<f32>) {
    normalize_pixel(pixel);
    linearize_pixel(pixel);
    apply_conversion_matrix_pixel(pixel);
}

fn apply_conversion_matrix_pixel(pixel: &mut Rgb<f32>) {
    let conversion_matrix: [[f32; 3]; 3] = [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ];

    // Matrix multiplication
    pixel.0 = [
        pixel.0[0] * conversion_matrix[0][0]
            + pixel.0[1] * conversion_matrix[0][1]
            + pixel.0[2] * conversion_matrix[0][2],
        pixel.0[0] * conversion_matrix[1][0]
            + pixel.0[1] * conversion_matrix[1][1]
            + pixel.0[2] * conversion_matrix[1][2],
        pixel.0[0] * conversion_matrix[2][0]
            + pixel.0[1] * conversion_matrix[2][1]
            + pixel.0[2] * conversion_matrix[2][2],
    ];
}

fn linearize_pixel(pixel: &mut Rgb<f32>) {
    fn is_less_than_eq_limit(color_comp: f32, limit: f32) -> bool {
        let epsilon = 1e-6;
        if (color_comp - limit).abs() <= epsilon {
            true
        } else if color_comp < limit {
            true
        } else {
            false
        }
    }

    if is_less_than_eq_limit(pixel.0[0], 0.04045) {
        pixel.0[0] /= 12.92;
    } else {
        pixel.0[0] = ((pixel.0[0] + 0.055) / 1.055).powf(2.4);
    }

    if is_less_than_eq_limit(pixel.0[1], 0.04045) {
        pixel.0[1] /= 12.92;
    } else {
        pixel.0[1] = ((pixel.0[1] + 0.055) / 1.055).powf(2.4);
    }

    if is_less_than_eq_limit(pixel.0[2], 0.04045) {
        pixel.0[2] /= 12.92;
    } else {
        pixel.0[2] = ((pixel.0[2] + 0.055) / 1.055).powf(2.4);
    }
}

fn normalize_pixel(pixel: &mut Rgb<f32>) {
    // Optimization: we only need to check the first pixel to see if normialzation is
    // even needed.
    if pixel.0[0] > 1. {
        // If not normalized, normalize
        pixel.0 = pixel.0.map(|value| value / 255.);
    } else {
        println!("Debug: Normalization for pixel not needed!");
    }
}

fn preprocess_image(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    convert_img_to_xyz(img);
    convert_xyz_to_lab(img);
}

// reference: http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
fn convert_xyz_to_lab(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    for pixel in img.pixels_mut() {
        pixelcolor_to_lab(pixel);
    }
}

fn convert_img_to_xyz(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    normalize_pixels_in_image(img);
    linearize_pixels_in_image(img);
    apply_conversion_matrix(img);
}

fn apply_conversion_matrix(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    // Matrix multiplication
    for pixel in img.pixels_mut() {
        apply_conversion_matrix_pixel(pixel);
    }
}

fn linearize_pixels_in_image(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    for pixel in img.pixels_mut() {
        linearize_pixel(pixel);
    }
}

fn normalize_pixels_in_image(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    let first_pixel = img.get_pixel(0, 0);

    // Optimization: we only need to check the first pixel to see if normialzation is
    // even needed.
    if first_pixel.0[0] > 1. {
        // If not normalized, normalize
        for pixel in img.pixels_mut() {
            normalize_pixel(pixel);
        }
    } else {
        println!("Debug: Normalization of pixels not needed!");
    }
}

fn color_distance(color1: Rgb<f32>, color2: Rgb<f32>) -> f32 {
    let r1 = color1.0[0];
    let g1 = color1.0[1];
    let b1 = color1.0[2];

    let r2 = color2.0[0];
    let g2 = color2.0[1];
    let b2 = color2.0[2];

    let r_diff_squared = r2.sub(r1).powf(2.);
    let g_diff_squared = g2.sub(g1).powf(2.);
    let b_diff_squared = b2.sub(b1).powf(2.);

    let diff_squared_sum = r_diff_squared + g_diff_squared + b_diff_squared;

    diff_squared_sum.sqrt()
}

fn pixelcolor_to_palettecolor(
    color1: Rgb<f32>,
    palette: &ColorPalette,
    lab_palette: &ColorPalette,
) -> Rgb<f32> {
    let mut current_smallest_idx = 0;
    let mut old_distance = f32::MAX;
    for (idx, palette_color) in lab_palette.palette.iter().enumerate() {
        let current_color_distance = color_distance(color1, *palette_color);

        if current_color_distance < old_distance {
            old_distance = current_color_distance;
            current_smallest_idx = idx;
        }
    }

    assert!(current_smallest_idx < palette.palette.len());
    palette.palette[current_smallest_idx]
}

fn rotate_img_by(img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>, angle: f32) {
    let img_copy = img.clone();
    for y in 0..img.height() - 1 {
        for x in 0..img.width() - 1 {
            let x = x as f32;
            let y = y as f32;

            println!("Rotating pixel ({x}, {y})");
            let x_rot = x * f32::cos(angle) - y * f32::sin(angle);
            let y_rot = x * f32::sin(angle) + y * f32::cos(angle);
            println!("x_rot: {x_rot}, y_rot: {y_rot}");

            let pixel = img_copy.get_pixel(x as u32, y as u32);
            img.put_pixel(x_rot as u32, y_rot as u32, *pixel);
        }
    }
}

// Using the Sobel Operator
fn edge_detect(img: &mut ImageBuffer<Luma<u8>, Vec<u8>>) {
    let g_x: [[f32; 3]; 3] = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]];
    let g_y: [[f32; 3]; 3] = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]];

    for y in 0..img.height() - 2 {
        for x in 0..img.width() - 2 {
            let val0 = img.get_pixel(y, x).0[0] as f32;
            let val1 = img.get_pixel(y.saturating_add(1), x).0[0] as f32;
            let val2 = img.get_pixel(y.saturating_add(2), x).0[0] as f32;
            let val3 = img.get_pixel(y, x.saturating_add(1)).0[0] as f32;
            let val5 = img.get_pixel(y + 2, x + 1).0[0] as f32;
            let val6 = img.get_pixel(y, x.saturating_add(2)).0[0] as f32;
            let val7 = img.get_pixel(y + 1, x + 2).0[0] as f32;
            let val8 = img.get_pixel(y + 2, x + 2).0[0] as f32;

            let gx = (-1. * val0) + (-2. * val3) + (-1. * val6) + val2 + (2. * val5) + val8;
            let gy = (-1. * val0) + (-2. * val1) + (-1. * val2) + val6 + (2. * val7) + val8;
            let mut mag = ((gx).powf(2.) + (gy).powf(2.)).sqrt();

            if mag > 255.0 {
                mag = 255.0;
            }

            let mag = mag as u8;

            img.put_pixel(y, x, Luma([mag]));
        }
    }
}

fn emboss(img: &mut ImageBuffer<Luma<u8>, Vec<u8>>) {
    let emboss_kernel = [[-1.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];

    for i in 1..img.height() - 1 {
        for j in 1..img.width() - 1 {
            let mut sum: f32 = 0.0;
            for ki in 0..3 {
                for kj in 0..3 {
                    let ki = ki as usize;
                    let kj = kj as usize;
                    let i = i as usize;

                    sum += img.get_pixel((i + ki - 1) as u32, j + kj as u32 - 1).0[0] as f32
                        * emboss_kernel[ki][kj] as f32;
                }
            }

            let sum = Luma([sum.clamp(0., 255.0) as u8]);
            img.put_pixel(i, j, sum);
        }
    }
}

fn sigma(width: u32, height: u32, blur_modifier: i32) -> f32 {
    return (((width * height) as f32) / 3630000.0) * blur_modifier as f32;
}

fn main() {
    // Parse the color_palette
    let color_palette = ColorPalette::new("color_palette.json");
    let lab_color_palette = color_palette.to_lab();

    println!("Lab Color Palette: {lab_color_palette:#?}");
    println!("Color Palette: {color_palette:#?}");

    let sigma = sigma(1054, 1054, 16);

    let cat_img = ImageReader::open("loki.png").unwrap().decode().unwrap();
    // .blur(sigma);

    let mut cat_imgbuf = cat_img.to_luma8();

    emboss(&mut cat_imgbuf);

    // Preprocess the image by converting to xyz
    let cat_imgbuf_copy = cat_imgbuf.clone();
    // preprocess_image(&mut cat_imgbuf);

    // let dithered_statue_imgbuf = floyd_steinberg_dither(statue_imgbuf);
    // let dithered_cat_imgbuf = floyd_steinberg_dither(
    //     cat_imgbuf_copy,
    //     &cat_imgbuf,
    //     &color_palette,
    //     &lab_color_palette,
    // );

    // Save the image dithered_cat_imgbuf
    image::save_buffer_with_format(
        "embossed_loki.png",
        cat_imgbuf.as_bytes(),
        cat_imgbuf.width(),
        cat_imgbuf.height(),
        image::ColorType::L8,
        image::ImageFormat::Png,
    )
    .unwrap();
}

fn floyd_steinberg_dither(
    mut original_img: ImageBuffer<Rgb<f32>, Vec<f32>>,
    img: &ImageBuffer<Rgb<f32>, Vec<f32>>,
    palette: &ColorPalette,
    lab_palette: &ColorPalette,
) -> ImageBuffer<Rgb<f32>, Vec<f32>> {
    for y in 0..img.height() - 1 {
        for x in 0..img.width() - 1 {
            let old_pixel = img.get_pixel(x, y).0;
            let new_pixel = {
                let old_pixel = img.get_pixel(x, y);
                find_closest_palette_color(old_pixel, palette, lab_palette)
            }
            .0;

            original_img.put_pixel(x, y, image::Rgb(new_pixel));

            let quant_error = sub_pixels(image::Rgb(old_pixel), image::Rgb(new_pixel));
            let quant_error = quant_error.0;
            // println!("{quant_error:?}");

            // println!("Old Pixel: {old_pixel:?}");
            pass1(x, y, quant_error, &mut original_img);
            let old_pixel = img.get_pixel(x, y).0;
            // println!("Pass 1: {old_pixel:?}");
            pass2(x, y, quant_error, &mut original_img);
            let old_pixel = img.get_pixel(x, y).0;
            // println!("Pass 2: {old_pixel:?}");
            pass3(x, y, quant_error, &mut original_img);
            let old_pixel = img.get_pixel(x, y).0;
            // println!("Pass 3: {old_pixel:?}");
            pass4(x, y, quant_error, &mut original_img);
            let old_pixel = img.get_pixel(x, y).0;
            // println!("Pass 4: {old_pixel:?}");
        }
    }

    original_img
}

fn sub_pixels(pixel1: Rgb<f32>, pixel2: Rgb<f32>) -> Rgb<f32> {
    let sub_pixel = pixel1.0.map(|pixel| pixel - pixel2.0[0]);

    image::Rgb(sub_pixel)
}

fn pass1(x: u32, y: u32, quant_error: [f32; 3], img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    // Scale quant_error
    let quant_error = scale_quant_error_by(quant_error, 7., 16.);
    // println!("quant_error: {quant_error:?}");

    // Add the quant_error to the pixel
    let quant_error_pixel = img
        .get_pixel(x.saturating_add(1), y)
        .0
        .map(|pixel| pixel + quant_error[0]);

    // println!("Quant Error Pixel: {quant_error_pixel:?}");

    // Store the pixel which was summed with the quant_error
    img.put_pixel(x + 1, y, image::Rgb(quant_error_pixel));
}

fn pass2(x: u32, y: u32, quant_error: [f32; 3], img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    // Scale quant_error
    let quant_error = scale_quant_error_by(quant_error, 3., 16.);

    // Add the quant_error to the pixel
    let quant_error_pixel = img
        .get_pixel(x.saturating_sub(1), y.saturating_add(1))
        .0
        .map(|pixel| pixel + quant_error[0]);

    // Store the pixel which was summed with the quant_error
    img.put_pixel(
        x.saturating_sub(1),
        y.saturating_add(1),
        image::Rgb(quant_error_pixel),
    );
}

fn pass3(x: u32, y: u32, quant_error: [f32; 3], img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    // Scale quant_error
    let quant_error = scale_quant_error_by(quant_error, 5., 16.);

    // Add the quant_error to the pixel
    let quant_error_pixel = img
        .get_pixel(x, y.saturating_add(1))
        .0
        .map(|pixel| pixel + quant_error[0]);

    // Store the pixel which was summed with the quant_error
    img.put_pixel(x, y + 1, image::Rgb(quant_error_pixel));
}

fn pass4(x: u32, y: u32, quant_error: [f32; 3], img: &mut ImageBuffer<Rgb<f32>, Vec<f32>>) {
    // Scale quant_error
    let quant_error = scale_quant_error_by(quant_error, 1., 16.);

    // Add the quant_error to the pixel
    let quant_error_pixel = img
        .get_pixel(x.saturating_add(1), y.saturating_add(1))
        .0
        .map(|pixel| pixel + quant_error[0]);

    // Store the pixel which was summed with the quant_error
    img.put_pixel(x + 1, y + 1, image::Rgb(quant_error_pixel));
}

fn scale_quant_error_by(quant_error: [f32; 3], numerator: f32, denominator: f32) -> [f32; 3] {
    let quant_error = quant_error.map(|pixel| pixel * (numerator / denominator));

    quant_error
}

fn find_closest_palette_color(
    old_pixel: &Rgb<f32>,
    palette: &ColorPalette,
    lab_palette: &ColorPalette,
) -> Rgb<f32> {
    pixelcolor_to_palettecolor(*old_pixel, palette, lab_palette)
}
