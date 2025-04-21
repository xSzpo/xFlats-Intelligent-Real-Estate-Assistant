output "ec2_chroma_role" {
  value = {
    arn  = aws_iam_role.ec2_chroma_role.arn
    id   = aws_iam_role.ec2_chroma_role.id
    name = aws_iam_role.ec2_chroma_role.name

  }
}

output "ec2_chroma_instance_profile" {
  value = {
    arn  = aws_iam_instance_profile.ec2_chroma_instance_profile.arn
    id   = aws_iam_instance_profile.ec2_chroma_instance_profile.id
    name = aws_iam_instance_profile.ec2_chroma_instance_profile.name
  }
}
