resource "aws_iam_role" "ec2_chroma_role" {
  name = "ec2_chroma_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "ec2.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_instance_profile" "ec2_chroma_instance_profile" {
  name = "chroma_instance_profile"
  role = aws_iam_role.ec2_chroma_role.name
}

resource "aws_iam_policy" "ec2_chroma_policy" {
  name        = "ec2_chroma_policy"
  description = "Policy for EC2 to access Secrets Manager and S3"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:ListAllMyBuckets"
        ],
        Resource = [
          "*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_chroma_policy_attach" {
  role       = aws_iam_role.ec2_chroma_role.name
  policy_arn = aws_iam_policy.ec2_chroma_policy.arn
}