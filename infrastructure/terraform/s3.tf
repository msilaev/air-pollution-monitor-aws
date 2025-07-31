resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "mlflow-artifacts-${random_id.suffix.hex}"
  force_destroy = true
  tags = {
    Name = "mlflow-artifacts"
  }
}

resource "random_id" "suffix" {
  byte_length = 4
}

output "mlflow_artifacts_bucket" {
  value = aws_s3_bucket.mlflow_artifacts.bucket
}
