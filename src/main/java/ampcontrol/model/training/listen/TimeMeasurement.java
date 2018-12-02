package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.LocalTime;
import java.util.function.BiConsumer;

/**
 * Measures time to complete one epoch of training.
 *
 * @author Christian Skärby
 */
public class TimeMeasurement extends BaseTrainingListener {

    private final static Logger log = LoggerFactory.getLogger(TimeMeasurement.class);

    private final BiConsumer<Integer, Double> measurementConsumer;

    private LocalTime startTime;
    private int nrofExamplesCount = 0;

    public TimeMeasurement() {
        this((nrofExamples, timeMs) ->
            log.info("Training took " +timeMs + " ms for " + nrofExamples + " examples, " + timeMs / nrofExamples + " ms per example"));
    }

    public TimeMeasurement(BiConsumer<Integer, Double> measurementConsumer) {
        this.measurementConsumer = measurementConsumer;
    }

    @Override
    public void onEpochStart(Model model) {
        startTime = LocalTime.now();
        nrofExamplesCount = 0;
    }

    @Override
    public void onEpochEnd(Model model) {
        final double endTimeMs = Duration.between(startTime, LocalTime.now()).toMillis();
        measurementConsumer.accept(nrofExamplesCount, endTimeMs);
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        nrofExamplesCount += model.batchSize();
    }
}
