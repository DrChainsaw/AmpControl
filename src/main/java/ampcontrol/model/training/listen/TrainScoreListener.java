package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

import java.util.function.BiConsumer;

/**
 * {@link BaseTrainingListener} which is basically a copy of {@link org.deeplearning4j.optimize.listeners.ScoreIterationListener}
 * with added capability of giving score and iteration to a provided {@link BiConsumer}. One exception towards the original
 * is that score is averaged over the reporting interval instead of being a snapshot.
 *
 * @author Christian Skärby
 */
public class TrainScoreListener extends BaseTrainingListener {

    private final BiConsumer<Integer, Double> iterAndScoreListener;
    private int iterCount = 0;
    private double resultSum = 0;
    private int lastIter;

    public TrainScoreListener(BiConsumer<Integer, Double> iterAndScoreListener) {
        this.iterAndScoreListener = iterAndScoreListener;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        resultSum += model.score();
        iterCount++;
        lastIter = iteration;
    }

    @Override
    public void onEpochEnd(Model model) {
        iterAndScoreListener.accept(lastIter, resultSum / Math.max(iterCount,1));
        iterCount = 0;
        resultSum = 0;
    }


}
