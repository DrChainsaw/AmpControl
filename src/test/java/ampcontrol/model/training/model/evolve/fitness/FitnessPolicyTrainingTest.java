package ampcontrol.model.training.model.evolve.fitness;

import ampcontrol.model.training.listen.MockModel;
import ampcontrol.model.training.model.ModelAdapter;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link FitnessPolicyTraining}
 *
 * @author Christian Skärby
 */
public class FitnessPolicyTrainingTest {

    /**
     * Test that fitness is applied correctly
     */
    @Test
    public void applyFitness() {
        final List<TrainingListener> listeners = new ArrayList<>();
        final MockModel model = new MockModel() {
            @Override
            public void setListeners(Collection<TrainingListener> listeners) {
                listeners = new ArrayList<>(listeners);
            }

            @Override
            public void addListeners(TrainingListener... listener) {
                 listeners.addAll(Stream.of(listener).collect(Collectors.toList()));
            }

            @Override
            public int numParams() {
                return (int) (3* 1e7);
            }

            @Override
            public double score() {
                return 1.2345678;
            }
        };
        final FitnessPolicy<ModelAdapter> policy = new FitnessPolicyTraining<>(3);
        final double[] measuredScore = {-1};

        policy.apply(new ModelAdapter() {
            @Override
            public void fit(DataSetIterator iter) {
                throw new UnsupportedOperationException("Not implemented!");
            }

            @Override
            public <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals) {
                throw new UnsupportedOperationException("Not implemented!");
            }

            @Override
            public Model asModel() {
                return model;
            }
        }, fitness -> measuredScore[0] = fitness);

        assertEquals("Incorrect number of training listeners", 2, listeners.size());

        // Bleh! Hardcoded knowledge of TraininListener implementation!
        listeners.forEach(listener -> listener.iterationDone(model, 0, 0));
        listeners.forEach(listener -> listener.onEpochEnd(model));
        assertEquals("No fitness shall have been reported!", -1d, measuredScore[0], 1e-10);

        listeners.forEach(listener -> listener.iterationDone(model, 1, 0));
        listeners.forEach(listener -> listener.onEpochEnd(model));
        assertEquals("No fitness shall have been reported!", -1d, measuredScore[0], 1e-10);

        listeners.forEach(listener -> listener.iterationDone(model, 2, 0));
        listeners.forEach(listener -> listener.onEpochEnd(model));
        assertEquals("No fitness shall have been reported!", 1.233, measuredScore[0], 1e-10);
    }
}