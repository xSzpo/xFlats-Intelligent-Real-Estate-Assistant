
resource "aws_secretsmanager_secret" "gemini" {
  name = "gemini-${data.aws_caller_identity.current.account_id}"
}

resource "aws_secretsmanager_secret_policy" "gemini_policy" {
  secret_arn = aws_secretsmanager_secret.gemini.arn

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid    = "AllowEC2AccessToGeminiSecret",
        Effect = "Allow",
        Principal = {
          AWS = [
          data.terraform_remote_state.iam.outputs.ec2_chroma_role.arn]
        },
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ],
        Resource = aws_secretsmanager_secret.gemini.arn
      }
    ]
  })
}
