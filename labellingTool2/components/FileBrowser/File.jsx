import { Card, CardBody, Tag } from "@chakra-ui/react";
import Link from "next/link";

export function File({ Key, LastModified, Size, Owner }) {
  const fileName = Key.split("/").slice(-1)[0];
  return (
    <Card>
      <CardBody>
        <Link href={`/view/${Key}`}>{fileName}</Link>
        <Tag>{Size}</Tag>
      </CardBody>
    </Card>
  );
}
