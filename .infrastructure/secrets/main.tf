provider "aws" {
  region  = "eu-central-1"
  profile = "priv"
}

terraform {
  backend "s3" {
    bucket  = "terraform-274181059559"
    key     = "state/secrets.tfstate"
    region  = "eu-central-1"
    encrypt = true
  }
}

