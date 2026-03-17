# test-dev-s3

This is a modified version of `test-dev` that loads images and masks from CloudFront URLs instead of MinIO.

## Key Differences from test-dev

### 1. Image Loading

- **test-dev**: Downloads images from MinIO using boto3 S3 client
- **test-dev-s3**: Downloads images directly from CloudFront URL using `requests`

### 2. Mask Loading

- **test-dev**: Downloads masks from MinIO bucket `ai-insurance/INSURANCE_RESULT/`
- **test-dev-s3**: Downloads masks from CloudFront URL `https://dyta7vmv7sqle.cloudfront.net/INSURANCE_RESULT/`

### 3. URL Construction

Images and masks are loaded using the CloudFront base URL:

```
Base URL: https://dyta7vmv7sqle.cloudfront.net/INSURANCE_RESULT/

Image URL example: https://dyta7vmv7sqle.cloudfront.net/INSURANCE_RESULT/uJNTHe-choF7ZFd_4J0C2u_EGJ0xMmJ.jpg
Mask URL example: https://dyta7vmv7sqle.cloudfront.net/INSURANCE_RESULT/NKKZRR07lS7vsqeAX-H_K.png
```

### 4. Dependencies

- **test-dev**: Requires `boto3` for MinIO access
- **test-dev-s3**: Only requires `requests` for HTTP downloads (no boto3 needed)

## Files Modified

### api_client.py

- Removed MinIO client initialization
- Changed `visualize_result()` call to not pass s3_client parameter
- Updated output path to `test-dev-s3/api_results.json`

### visualize_masks.py

- Removed `get_minio_client()` function
- Removed `download_from_minio()` function
- Added `download_from_url()` function to download from CloudFront
- Added `download_image_from_url()` function to download images directly into memory
- Updated `visualize_result()` to:
  - Not require s3_client parameter
  - Construct CloudFront URLs using `CLOUDFRONT_BASE_URL`
  - Download images from `imageResize` field using CloudFront URL
  - Download masks from `maskPath` field using CloudFront URL
- Updated paths to use `test-dev-s3` directories

### test_api_client.sh

- Updated output paths in messages

## Usage

Same as test-dev:

```bash
# Single image
python api_client.py image1.jpg

# Multiple images
python api_client.py image1.jpg image2.jpg image3.jpg

# Or use the test script
./test_api_client.sh
```

## Output

- Visualizations: `/home/a4000/Data/ducanhng/CV/Chore/test-dev-s3/visualization/`
- API Results: `/home/a4000/Data/ducanhng/CV/Chore/test-dev-s3/api_results.json`
- Temp downloads: `/tmp/mask_download_s3/`
