package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.LocalTime;
import java.util.List;
import java.util.Map;

/**
 * Measures time to complete one epoch of training.
 *
 * @author Christian Skärby
 */
public class TimeMeasurement implements TrainingListener {

    private static final Logger log = LoggerFactory.getLogger(TimeMeasurement.class);

    private LocalTime startTime;
    private int nrofExamplesCount = 0;

    @Override
    public void onEpochStart(Model model) {
        startTime = LocalTime.now();
        nrofExamplesCount = 0;
    }

    @Override
    public void onEpochEnd(Model model) {
        final double endTimeMs = Duration.between(startTime, LocalTime.now()).toMillis();
        log.info("Training took " + endTimeMs + " ms for " + nrofExamplesCount + " examples, " + endTimeMs / nrofExamplesCount + " ms per example");
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {

    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {

    }

    @Override
    public void onGradientCalculation(Model model) {

    }

    @Override
    public void onBackwardPass(Model model) {

    }

    private boolean invoked = false;

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        invoke();
        nrofExamplesCount += model.batchSize();
    }
}
