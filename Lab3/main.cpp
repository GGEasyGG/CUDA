#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

using namespace std;

__constant__ double3 avg_gpu[32];

__global__ void min_dist_kernel(const int width, const int height, uchar4 *image_gpu, uchar4 *image_out, int nc)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    uchar4 pixel;
    double3 avg;

    double maximum;
    double sum;
    int arg;

    while(index < width * height)
    {
        pixel = image_gpu[index];

        for(int j = 0; j < nc; j++)
        {
            avg = avg_gpu[j];

            sum = -(pixel.x - avg.x) * (pixel.x - avg.x) - (pixel.y - avg.y) * (pixel.y - avg.y) - (pixel.z - avg.z) * (pixel.z - avg.z);

            if(j == 0)
            {
                maximum = sum;
                arg = j;
            }
            else
            {
                if(sum > maximum)
                {
                    maximum = sum;
                    arg = j;
                }
            }
        }

        image_out[index] = make_uchar4(pixel.x, pixel.y, pixel.z, arg);

        index += offset;
    }
}

int main()
{
    string input_path, output_path;

    cin >> input_path;
    cin >> output_path;

    char *input = new char[input_path.length() + 1];
    strcpy(input, input_path.c_str());

    char *output = new char[output_path.length() + 1];
    strcpy(output, output_path.c_str());

    int width, height;

    FILE *input_file = fopen(input, "rb");
    fread(&width, sizeof(int), 1, input_file);
    fread(&height, sizeof(int), 1, input_file);

    uchar4 *image = (uchar4 *)malloc(sizeof(uchar4) * width * height);

    fread(image, sizeof(uchar4), width * height, input_file);
    fclose(input_file);

    uchar4 *image_gpu;

    cudaMalloc(&image_gpu, width * height * sizeof(uchar4));
    cudaMemcpy(image_gpu, image, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);

    int nc, np_j;

    cin >> nc;

    uchar4 pixel;
    double p_x, p_y, p_z;

    int x, y;

    double3 *avg = new double3[nc];

    for(int j = 0; j < nc; j++)
    {
        cin >> np_j;

        p_x = 0.0;
        p_y = 0.0;
        p_z = 0.0;

        for(int i = 0;  i < np_j; i++)
        {
            cin >> x >> y;

            pixel = image[y * width + x];

            p_x += pixel.x;
            p_y += pixel.y;
            p_z += pixel.z;
        }

        p_x /= np_j;
        p_y /= np_j;
        p_z /= np_j;

        avg[j] = make_double3(p_x, p_y, p_z);
    }

    cudaMemcpyToSymbol(avg_gpu, avg, nc * sizeof(double3), 0, cudaMemcpyHostToDevice);

    uchar4 *image_out;
    cudaMalloc(&image_out, sizeof(uchar4) * width * height);

    uchar4 *new_image = (uchar4 *)malloc(sizeof(uchar4) * width * height);

    min_dist_kernel<<< 256, 256 >>>(width, height, image_gpu, image_out, nc);

    cudaMemcpy(new_image, image_out, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

    cudaFree(image_out);
    cudaFree(image_gpu);

    FILE *output_file = fopen(output, "wb");
    fwrite(&width, sizeof(int), 1, output_file);
    fwrite(&height, sizeof(int), 1, output_file);
    fwrite(new_image, sizeof(uchar4), width * height, output_file);
    fclose(output_file);

    free(image);
    free(new_image);

    delete []avg;
    delete []input;
    delete []output;

    return 0;
}
