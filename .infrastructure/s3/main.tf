provider "aws" {
  region  = "eu-central-1"
  profile = "priv"
}

terraform {
  backend "s3" {
    bucket  = "terraform-274181059559"
    key     = "state/s3.tfstate"
    region  = "eu-central-1"
    encrypt = true
  }
}

resource "aws_s3_bucket" "terraform" {
  bucket = "terraform-${data.aws_caller_identity.current.account_id}"

  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_s3_bucket" "data" {
  bucket = "data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.data.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          AWS = [
            data.terraform_remote_state.iam.outputs.ec2_chroma_role.arn
          ]
        },
        Action = [
          "s3:GetObject",
          "s3:GetObjectAttributes",
          "s3:GetBucketLocation",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:ListMultipartUploadParts",
          "s3:GetEncryptionConfiguration",
          "s3:ListBucketVersions",
          "s3:GetObjectAcl",
          "s3:GetObjectTagging",
          "s3:GetObjectVersion",
          "s3:PutObjectAcl",
          "s3:PutObject",
          "s3:AbortMultipartUpload",
          "s3:PutObjectTagging",
          "s3:DeleteObject"
        ],
        Resource = [
          "arn:aws:s3:::${aws_s3_bucket.data.id}",
          "arn:aws:s3:::${aws_s3_bucket.data.id}/*"
        ]
      }
    ]
  })
}
