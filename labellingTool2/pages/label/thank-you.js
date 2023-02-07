import { Heading, Link } from "@chakra-ui/react";
import Confetti from 'react-confetti';
import useWindowSize from 'react-use/lib/useWindowSize';
import { useRouter } from "next/router";
export default function ThankYou(props){
const router = useRouter()
const { run } = router.query;
const { width, height } = useWindowSize(1920, 1080);

    return (
        <>
            <Heading>Thank you!</Heading>
            <Link href={`/label/${run}`}>Label more</Link>
            <Confetti width={width} height={height}/>
        </>
    );
}