import { LambdaClient, AddLayerVersionPermissionCommand } from "@aws-sdk/client-lambda";

module.exports.init = function () {
    const client = new LambdaClient({ region: "REGION" });
  };
