import { Card, CardBody, Tag } from "@chakra-ui/react";
import Link from "next/link";

export function Dir({ Key }) {
  return (
    <Card>
      <CardBody>
        <Link href={`/browse/${Key}`}>DIR: {Key}</Link>
      </CardBody>
    </Card>
  );
}
