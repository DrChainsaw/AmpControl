package ampControl.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.function.BiConsumer;

/**
 * {@link IterationListener} which is basically a copy of {@link org.deeplearning4j.optimize.listeners.ScoreIterationListener}
 * with added capability of giving score and iteration to a provided {@link BiConsumer}. One exception towards the original
 * is that score is averaged over the reporting interval instead of being a snapshot.
 *
 * @author Christian Skärby
 */
public class TrainScoreListener implements IterationListener {

    private final BiConsumer<Integer, Double> iterAndScoreListener;
    private final int printIterations;
    private int iterCount = 0;
    private double resultSum = 0;

    public TrainScoreListener(int printIterations, BiConsumer<Integer, Double> iterAndScoreListener) {
       this.printIterations = printIterations;
        this.iterAndScoreListener = iterAndScoreListener;
    }

    private boolean invoked = false;

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {

        resultSum += model.score();
        if (iterCount % printIterations == 0) {
            iterAndScoreListener.accept(iteration, resultSum / Math.max(iterCount,1));
            iterCount = 0;
            resultSum = 0;
        }
        iterCount++;
    }
}
