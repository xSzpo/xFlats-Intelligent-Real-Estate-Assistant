data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "terraform_remote_state" "iam" {
  backend = "s3"
  config = {
    bucket = "terraform-011337673661"
    key    = "state/iam.tfstate"
    region = "eu-central-1"
  }
}
