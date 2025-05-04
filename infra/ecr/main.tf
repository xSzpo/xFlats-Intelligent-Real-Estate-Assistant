provider "aws" {
  region  = "eu-central-1"
  profile = "priv"
}

terraform {
  backend "s3" {
    bucket  = "terraform-274181059559"
    key     = "state/ecr.tfstate"
    region  = "eu-central-1"
    encrypt = true
  }
}


resource "aws_ecr_repository" "xflats_crawler" {
  name = "xflats-crawler"
}

resource "aws_ecr_lifecycle_policy" "xflats_crawler" {
  repository = aws_ecr_repository.xflats_crawler.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only 10 images"
        selection = {
          tagStatus   = "untagged"
          countType   = "imageCountMoreThan"
          countNumber = 30
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}