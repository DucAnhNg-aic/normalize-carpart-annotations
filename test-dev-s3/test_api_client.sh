#!/bin/bash
# Example script to test API client with sample images (S3 version)

echo "=================================="
echo "API Client Test Script (S3)"
echo "=================================="
echo ""

API_IP="192.168.80.25"

export API_IP=$API_IP
python api_client.py  "DA_TEST/image_sensitive_case.png"
echo ""
echo "=================================="
echo "Test completed!"
echo "Check results in:"
echo "  - Visualizations: /home/a4000/Data/ducanhng/CV/Chore/test-dev-s3/visualization/"
echo "  - API Results: api_results.json"
echo "=================================="
