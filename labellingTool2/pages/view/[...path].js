import {
    getAndReadStringObject
} from "../../utils/aws/s3";
import { Labeller } from "../../components/Labeller";
import { FlagsProvider } from "flagged";
import { localLabelling } from "../../featureFlags";

function Page({ reviews, sampleFileName }) {
    const features = localLabelling();

    return (
        <FlagsProvider features={features}>
            <Labeller reviews={reviews} />
        </FlagsProvider>
    );
}

export async function getServerSideProps(context) {
    const Key = context.params.path.join("/");

    const data = JSON.parse(await getAndReadStringObject(Key));
    return { props: { ...data } };
}

export default Page;
