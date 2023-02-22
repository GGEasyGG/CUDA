#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>

using namespace std;

texture<uchar4, 2, cudaReadModelElementType> tex;
texture<double4, 2, cudaReadModelElementType> tex_final;

const double PI = 3.141592653589793;

__global__ void gauss_kernel_horizontal(const int width, const int height, double4 *image_out, int r, const double *pattern_gpu)
{
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    int x, y, radius;
    uchar4 pixel;
    uchar4 pixel_tmp;

    double p_x, p_y, p_z;

    for(y = index_y; y < height; y += offset_y)
    {
        for(x = index_x; x < width; x += offset_x)
        {
            pixel = tex2D(tex, x, y);

            p_x = 0.0;
            p_y = 0.0;
            p_z = 0.0;

            for(radius = -r; radius <= r; radius++)
            {
                if(radius == 0)
                {
                    p_x += pixel.x * pattern_gpu[radius];
                    p_y += pixel.y * pattern_gpu[radius];
                    p_z += pixel.z * pattern_gpu[radius];
                }
                else
                {
                    pixel_tmp = tex2D(tex, x + radius, y);

                    p_x += pixel_tmp.x * pattern_gpu[radius];
                    p_y += pixel_tmp.y * pattern_gpu[radius];
                    p_z += pixel_tmp.z * pattern_gpu[radius];
                }

            }

            p_x /= (r * sqrt(2 * PI));
            p_y /= (r * sqrt(2 * PI));
            p_z /= (r * sqrt(2 * PI));

            image_out[y * width + x] = make_double4(p_x, p_y, p_z, pixel.w);
        }
    }
}

__global__ void gauss_kernel_vertical(const int width, const int height, uchar4 *image_out_final, int r, const double *pattern_gpu)
{
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    int x, y, radius;
    double4 pixel;
    double4 pixel_tmp;

    double p_x, p_y, p_z;

    for(y = index_y; y < height; y += offset_y)
    {
        for(x = index_x; x < width; x += offset_x)
        {
            pixel = tex2D(tex_final, x, y);

            p_x = 0.0;
            p_y = 0.0;
            p_z = 0.0;

            for(radius = -r; radius <= r; radius++)
            {
                if(radius == 0)
                {
                    p_x += pixel.x * pattern_gpu[radius];
                    p_y += pixel.y * pattern_gpu[radius];
                    p_z += pixel.z * pattern_gpu[radius];
                }
                else
                {
                    pixel_tmp = tex2D(tex, x, y + radius);

                    p_x += pixel_tmp.x * pattern_gpu[radius];
                    p_y += pixel_tmp.y * pattern_gpu[radius];
                    p_z += pixel_tmp.z * pattern_gpu[radius];
                }

            }

            p_x /= (r * sqrt(2 * PI));
            p_y /= (r * sqrt(2 * PI));
            p_z /= (r * sqrt(2 * PI));

            image_out_final[y * width + x] = make_uchar4(p_x, p_y, p_z, pixel.w);
        }
    }
}

int main()
{
    string input_path, output_path;

    cin >> input_path;
    cin >> output_path;

    int r;

    cin >> r;

    int width, height;

    FILE *input_file = fopen(input_path, "rb");
    fread(&width, sizeof(int), 1, input_file);
    fread(&height, sizeof(int), 1, input_file);

    uchar4 *image = (uchar4 *)malloc(sizeof(uchar4) * width * height);

    fread(image, sizeof(uchar4), width * height, input_file);
    fclose(input_file);

    cudaArray *image_array;
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&image_array, &channel, width, height);

    cudaMemcpy2DToArray(image_array, 0, 0, image, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice);

    tex.normalized = false;
    tex.channelDesc = channel;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex, image_array, channel);

    double4 *image_out;
    cudaMalloc(image_out, sizeof(double4) * width * height);

    double pattern[2 * r + 1];

    double sum = 0.0;

    for(int i = -r; i <= r; i++)
    {
        pattern[i + r] = exp(-(i * i) / (2.0 * r * r));
        sum += pattern[i + r];
    }

    for(int i = -r; i <= r; i++)
    {
        pattern[i + r] /= sum;
    }

    __constant__ double pattern_gpu[2 * r + 1];

    cudaMemcpyToSymbol(pattern_gpu, pattern, 2 * r + 1, 0, cudaMemcpyHostToDevice);

    double4 *new_image = (double4 *)malloc(sizeof(double4) * width * height);

    gauss_kernel_horizontal<<< dim3(16, 16), dim3(32, 32) >>>(width, height, image_out, r, pattern_gpu);

    cudaMemcpy(new_image, image_out, sizeof(double4) * width * height, cudaMemcpyDeviceToHost);
    cudaUnbindTexture(tex);
    cudaFreeArray(image_array);
    cudaFree(image_out);

    cudaArray *image_array_final;
    cudaChannelFormatDesc channel_final = cudaCreateChannelDesc<double4>();
    cudaMallocArray(&image_array_final, &channel_final, width, height);

    cudaMemcpy2DToArray(image_array_final, 0, 0, new_image, width * sizeof(double4), width * sizeof(double4), height, cudaMemcpyHostToDevice);

    tex_final.normalized = false;
    tex_final.channelDesc = channel;
    tex_final.addressMode[0] = cudaAddressModeClamp;
    tex_final.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(tex_final, image_array_final, channel_final);

    uchar4 *image_out_final;
    cudaMalloc(image_out_final, sizeof(uchar4) * width * height);

    uchar4 *new_image_final = (uchar4 *)malloc(sizeof(uchar4) * width * height);

    gauss_kernel_vertical<<< dim3(16, 16), dim3(32, 32) >>>(width, height, image_out_final, r, pattern_gpu);

    cudaMemcpy(new_image_final, image_out_final, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);
    cudaUnbindTexture(tex_final);
    cudaFreeArray(image_array_final);
    cudaFree(image_out_final);

    cudaFree(pattern_gpu);

    FILE *output_file = fopen(output_path, "wb");
    fwrite(&width, sizeof(int), 1, output_file);
    fwrite(&height, sizeof(int), 1, output_file);
    fwrite(new_image_final, sizeof(uchar4), width * height, output_file);
    fclose(output_file);

    free(image);
    free(new_image);
    free(new_image_final);

    return 0;
}
