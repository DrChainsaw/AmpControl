package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.LocalTime;

/**
 * Measures time to complete one epoch of training.
 *
 * @author Christian Skärby
 */
public class TimeMeasurement extends BaseTrainingListener {

    private static final Logger log = LoggerFactory.getLogger(TimeMeasurement.class);

    private LocalTime startTime;
    private int nrofExamplesCount = 0;
    private int lastEpoch = -1;

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
    public void iterationDone(Model model, int iteration, int epoch) {
        if(epoch > lastEpoch) {
            if(lastEpoch > -1) {
                onEpochEnd(model);
            }
            onEpochStart(model);
            lastEpoch = epoch;
        }
        nrofExamplesCount += model.batchSize();
    }
}
