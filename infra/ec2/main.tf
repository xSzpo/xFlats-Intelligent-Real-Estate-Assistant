provider "aws" {
  region  = "eu-central-1"
  profile = "priv"
}

terraform {
  backend "s3" {
    bucket  = "terraform-011337673661"
    key     = "state/ec2.tfstate"
    region  = "eu-central-1"
    encrypt = true
  }
}

