package ampControl.amp;

import com.beust.jcommander.Parameters;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Self explanatory. Used when no action from a classification is wanted
 *
 * @author Christian Skärby
 */
@Parameters(commandDescription = "Dummy listener")
public class DummyClassifictionListener implements  ClassificationListener {

    @Override
    public void indicateAudioClassification(INDArray probabilities) {
        //Ignore
    }
}
