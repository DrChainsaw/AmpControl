package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * {@link IterationListener} which is basically a copy of {@link org.deeplearning4j.optimize.listeners.ScoreIterationListener}
 * with added capability of giving score and iteration to a provided {@link BiConsumer}. One exception towards the original
 * is that score is averaged over the reporting interval instead of being a snapshot.
 *
 * @author Christian Skärby
 */
public class TrainScoreListener implements TrainingListener {

    private final BiConsumer<Integer, Double> iterAndScoreListener;
    private int iterCount = 0;
    private double resultSum = 0;
    private int lastIter;

    public TrainScoreListener(BiConsumer<Integer, Double> iterAndScoreListener) {
        this.iterAndScoreListener = iterAndScoreListener;
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
        resultSum += model.score();
        iterCount++;
        lastIter = iteration;
    }

    @Override
    public void onEpochStart(Model model) {

    }

    @Override
    public void onEpochEnd(Model model) {
        iterAndScoreListener.accept(lastIter, resultSum / Math.max(iterCount,1));
        iterCount = 0;
        resultSum = 0;
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
}
