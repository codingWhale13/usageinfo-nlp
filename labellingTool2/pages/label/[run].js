import {
  listObjects,
  getAndReadStringObject,
  moveObject,
  DYNAMIC_SAMPLING_BASE_FOLDER,
} from "../../utils/aws/s3";
import { Labeller } from "../../components/Labeller";
import { FlagsProvider } from "flagged";
import { labelUsageOptionsDynamic } from "../../featureFlags";
import { useRouter } from "next/router";
import { formatJsonReviews } from "../../utils/files";
function Page({ reviews, sampleFileName }) {
  const features = labelUsageOptionsDynamic();
  const router = useRouter();
  const { run } = router.query;

  return (
    <FlagsProvider features={features}>
      <Labeller reviews={reviews} run={run} sampleFileName={sampleFileName} />
    </FlagsProvider>
  );
}

// This gets called on every request
export async function getServerSideProps(context) {
  const runFolder = DYNAMIC_SAMPLING_BASE_FOLDER + context.params.run;
  const sourceFolder = runFolder + "/backlog/";
  const inProgressFolder = runFolder + "/in_progress/";

  const availableSamples = (await listObjects(sourceFolder)).filter((file) =>
    file.Key.endsWith(".json")
  );

  if (availableSamples.length > 0) {
    const choosenSample =
      availableSamples[Math.floor(Math.random() * availableSamples.length)];
    const choosenSampleFileName = choosenSample.Key.split("/").slice(-1)[0];
    console.log(
      `Choose ${choosenSample.Key} from ${availableSamples.length} available samples.`
    );

    let choosenSampleReviews = JSON.parse(
      await getAndReadStringObject(choosenSample.Key)
    );
    await moveObject(
      choosenSample.Key,
      inProgressFolder + choosenSampleFileName
    );
    return {
      props: {
        ...formatJsonReviews(choosenSampleReviews),
        sampleFileName: choosenSampleFileName,
      },
    };
  } else {
    console.log("No samples available");
    return { props: { reviews: [] } };
  }
}

export default Page;
