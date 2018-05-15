package ampControl.model.training;

import ampControl.model.training.listen.MockModel;
import ampControl.model.training.model.ModelHandle;
import ampControl.model.training.model.validation.Validation;
import ampControl.model.visualize.Plot;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.function.BiConsumer;
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

        final TrainingHarness harness = new TrainingHarness(new ArrayList<>(models), "", plotFac, path -> str -> {/* ignore */});
        final int nrofTrainingSteps = 100;
        harness.startTraining(nrofTrainingSteps);

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
        private final Set<String> savedModelNames = new HashSet<>();

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

        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public double getBestEvalScore() {
            return 0;
        }

        @Override
        public Model getModel() {
            return model;
        }

        @Override
        public int getNrofBatchesForTraining() {
            return 0;
        }

        @Override
        public int getNrofTrainingExamplesPerBatch() {
            return 0;
        }

        @Override
        public int getNrofEvalExamples() {
            return 0;
        }

        @Override
        public void createTrainingEvalListener(BiConsumer<Integer, Double> accuracyCallback) {

        }

        @Override
        public void registerValidation(Validation.Factory<? extends IEvaluation> validationFactory) {
            validations.add(validationFactory.create(labels));
        }

        @Override
        public void saveModel(String fileName) {
            savedModelNames.add(fileName);
        }
    }

    private static class MockPlot implements Plot<Integer, Double> {


        @Override
        public void createSeries(String label) {

        }

        @Override
        public void plotData(String label, Integer integer, Double aDouble) {

        }

        @Override
        public void storePlotData(String label) {

        }
    }
}