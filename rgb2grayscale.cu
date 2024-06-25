#include <stdio.h>
#include <assert.h>

#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// TODO: CUDA definitions
__global__
void rgb2gray_kernel(unsigned char *in, unsigned char *out, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size) {
        out[x] = (unsigned char)floor(0.2126*in[x*3] + 0.7152*in[x*3+1] + 0.0722*in[x*3+2]);
    }
}

void handler(unsigned char * in_h, unsigned char * out_h, int w, int h) {
    const int size2D = w * h * sizeof(unsigned char);      // ya, I know it's 1...
    const int size3D = size2D * 3;
    unsigned char *in_d, *out_d;

    cudaMalloc((void **)&in_d, size3D);
    cudaMalloc((void **)&out_d, size2D);

    cudaMemcpy(in_d, in_h, size3D, cudaMemcpyHostToDevice);

    // TODO: kernel call
    rgb2gray_kernel<<<ceil(size2D/256.0), 256>>>(in_d, out_d, size2D);    // !

    cudaMemcpy(out_h, out_d, size2D, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);
}

int main(int argc, char **argv) {
    int input_w, input_h, input_c, output_w, output_h;

    if (argc == 1) {
        printf("Usage: %s <RGB_input_image> [output_width] [output_height]\n", argv[0]);
        return 1;
    }
    unsigned char * img = stbi_load(argv[1], &input_w, &input_h, &input_c, 0);
    unsigned char * resized_img, * out;
    
    if (img == NULL) {
        printf("Error loading image\n");
        return 1;
    }

    printf("image selected: %s\n", argv[1]);
    printf("image shape: (%d, %d, %d)\n", input_w, input_h, input_c);

    if (input_c == 1) {
        printf("Already grayscale!\nFalling back!\n");
        return 1;
    }
    assert (input_c == 3);

    output_w = input_w; output_h = input_h;
    switch (argc) {
        case 4:
            output_h = atoi(argv[3]);
        case 3:
            output_w = atoi(argv[2]);
            break;
    }

    // resize if needed
    if (!(argc == 2 || input_w == output_w && input_h == output_h)) {
        resized_img = stbir_resize_uint8_srgb(img, input_w, input_h, 0, 
                                            NULL, output_w, output_h, 0,
                                            (stbir_pixel_layout)input_c);

        if (resized_img == NULL) {
            printf("Error resizing image\n");
            stbi_image_free(img);
            return 1;
        }
        stbi_image_free(img);
        img = resized_img;

        printf("reshaped image: (%d, %d, %d)\n", output_w, output_h, input_c);
    }

    out = (unsigned char *)malloc(output_w * output_h * sizeof(unsigned char));
    if (out == NULL) {
        printf("Error allocating memory for output\n");
        stbi_image_free(img);
        return 1;
    }

    // TODO: CUDA calls
    handler(img, out, output_w, output_h);

    // TODO: Export
    stbi_write_jpg("./out.jpg", output_w, output_h, 1, out, 100);

    stbi_image_free(img);
    free(out);

    return 0;
}