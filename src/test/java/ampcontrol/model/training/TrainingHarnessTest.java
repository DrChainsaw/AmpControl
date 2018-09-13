package ampcontrol.model.training;

import ampcontrol.model.training.listen.MockModel;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.naming.AddPrefix;
import ampcontrol.model.training.model.validation.Validation;
import ampcontrol.model.visualize.Plot;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link TrainingHarness}
 *
 * @author Christian Skärby
 */
public class TrainingHarnessTest {

    /**
     * Test model training. Basically just a smoke test.
     */
    @Test
    public void training() {
        final List<ProbeModelHandle> models = IntStream.range(0, 3)
                .mapToObj(i -> new ProbeModelHandle("ProbeModel" + i))
                .collect(Collectors.toList());
        final Plot.Factory<Integer, Double> plotFac = title -> new MockPlot();

        final TrainingHarness harness = new TrainingHarness(new ArrayList<>(models), new AddPrefix("dummy_"), plotFac, path -> str -> {/* ignore */});
        final int nrofTrainingSteps = 300;
        harness.startTraining(nrofTrainingSteps);

        // Attempt to prevent weird failures in CI by giving other threads time to finish
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            // Should not matter
        }

        models.forEach(model -> model.assertNrofFitCalls(nrofTrainingSteps));
        models.forEach(model -> model.assertNrofEvalCalls(nrofTrainingSteps));
        models.forEach(model -> model.assertNrofSaveFileNames(2));


    }

    private static class ProbeModelHandle implements ModelHandle {

        private int nrofFitCalls = 0;
        private int nrofEvalCalls = 0;
        private final String name;

        private final Model model = new MockModel();
        private final Collection<Validation<? extends IEvaluation>> validations = new ArrayList<>();
        private final List<String> labels = Arrays.asList("greg", "grgrhh");
        private final Set<String> savedModelNames = new LinkedHashSet<>();

        private ProbeModelHandle(String name) {
            this.name = name;
        }

        @Override
        public void fit() {
            nrofFitCalls++;
        }

        @Override
        public void eval() {
            final double[] result = new double[labels.size()];
            IntStream.range(0, nrofEvalCalls).limit(labels.size()).forEach(i -> result[i] = 1);

            nrofEvalCalls++;
            validations.stream().map(Validation::get)
                    .forEach(ieOpt -> ieOpt.ifPresent(ie -> ie.eval(Nd4j.create(result), Nd4j.zeros(labels.size()))));
            validations.forEach(Validation::notifyComplete);
        }

        private void assertNrofFitCalls(int expected) {
            assertEquals("Incorrect number of fit calls!", expected, nrofFitCalls);
        }

        private void assertNrofEvalCalls(int expected) {
            assertEquals("Incorrect number of fit calls!", expected, nrofEvalCalls);
        }

        private void assertNrofSaveFileNames(int expected) {
            assertEquals("Incorrect number of save model names: " + savedModelNames, expected, savedModelNames.size());
        }

        @Override
        public void resetTraining() {
            //Ignore
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public Model getModel() {
            return model;
        }

        @Override
        public void registerValidation(Validation.Factory<? extends IEvaluation> validationFactory) {
            validations.add(validationFactory.create(labels));
        }

        @Override
        public synchronized void saveModel(String fileName) {
            savedModelNames.add(fileName);
        }
    }

    private static class MockPlot implements Plot<Integer, Double> {


        @Override
        public void createSeries(String label) {
            //Ignore
        }

        @Override
        public void plotData(String label, Integer integer, Double aDouble) {
            //Ignore
        }

        @Override
        public void storePlotData(String label) {
            //Ignore
        }
    }
}