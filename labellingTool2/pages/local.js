import { FlagsProvider } from 'flagged';
import { localLabelling } from '../featureFlags';
import { Labeller } from '../components/Labeller';

function App(){
  return (
    <FlagsProvider features={localLabelling()}>
        <Labeller />
    </FlagsProvider>
  );
}

export default App;