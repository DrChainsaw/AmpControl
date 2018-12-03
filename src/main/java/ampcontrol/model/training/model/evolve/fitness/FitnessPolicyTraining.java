package ampcontrol.model.training.model.evolve.fitness;

import ampcontrol.model.training.listen.TrainScoreListener;
import ampcontrol.model.training.model.ModelAdapter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Adds scores fitness based on training performance as well as number of parameters in model.
 *
 * @author Christian Skärby
 */
public class FitnessPolicyTraining<T extends ModelAdapter> implements FitnessPolicy<T> {

    private static final Logger log = LoggerFactory.getLogger(FitnessPolicyTraining.class);

    private final int nrofItersToAccumulate;

    public FitnessPolicyTraining(int nrofItersToAccumulate) {
        this.nrofItersToAccumulate = nrofItersToAccumulate;
    }


    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {

        final BiConsumer<Integer, Double> fitnessCalculation = new BiConsumer<Integer, Double>() {
            int cnt = 0;
            double scoreSum = 0;

            @Override
            public void accept(Integer iter, Double score) {
                scoreSum += score;
                cnt++;
                if (cnt == nrofItersToAccumulate) {
                    final double scoreContrib = Math.round(scoreSum * 1e3 / cnt) / 1e3;
                    final double paramContrib = candidate.asModel().numParams() / 1e10;
                    log.info("score: " + (scoreSum / cnt) + " contrib: " + scoreContrib);
                    log.info("params: " + candidate.asModel().numParams() + " contrib: " + paramContrib);

                    fitnessListener.accept(
                            scoreContrib + paramContrib);
                    cnt = 0;
                }
            }
        };
        candidate.asModel().addListeners(new TrainScoreListener(fitnessCalculation));
        return candidate;
    }
}
