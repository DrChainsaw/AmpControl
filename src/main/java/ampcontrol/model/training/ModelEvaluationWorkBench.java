package ampcontrol.model.training;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.listen.ActivationContribution;
import ampcontrol.model.training.model.EvolvingGraphAdapter;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelAdapter;
import ampcontrol.model.training.model.evolve.mutate.MutateLayerContained;
import ampcontrol.model.training.model.evolve.mutate.MutateNout;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * Evaluates a model. Main use case is to test effects of various
 * {@link ampcontrol.model.training.model.evolve.mutate.Mutation}s.
 *
 * @author Christian Skärby
 */
class ModelEvaluationWorkBench {

    private static final Logger log = LoggerFactory.getLogger(TrainingDescription.class);

    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;

    ModelEvaluationWorkBench(
            MiniEpochDataSetIterator trainIter,
            MiniEpochDataSetIterator evalIter) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
    }

    public void evalute(final ComputationGraph graph, final String layer) {

        evalBaseline(graph);

        evalIncreaseNout(graph, layer);

        evalDecreaseNout(graph, layer);

        evalDecreaseNoutOptimal(graph, layer);

        evalIncreaseKernelSize(graph, layer);

        evalDecreaseKernelSize(graph, layer);
    }

    private void evalDecreaseKernelSize(ComputationGraph graph, String layer) {
        final String experimentLabel = "decreaseKernelSize: ";
        final ModelAdapter decreaseKs = new EvolvingGraphAdapter(graph, new MutateLayerContained(() -> Stream.of(MutateLayerContained.LayerMutation.builder()
                .layerName(layer)
                .inputLayers(getInputLayers(graph, layer))
                .mutation(layerConfOpt -> layerConfOpt
                        .map(Layer::clone)
                        .filter(layerConf -> layerConf instanceof ConvolutionLayer)
                        .map(layerConf -> (ConvolutionLayer) layerConf)
                        .map(convConf -> {
                            convConf.setKernelSize(new int[]{convConf.getKernelSize()[0] - 1, convConf.getKernelSize()[1]});
                            convConf.setWeightInit(WeightInit.DISTRIBUTION);
                            convConf.setDist(new ConstantDistribution(0));
                            return convConf;
                        })
                        .orElseThrow(() -> new IllegalArgumentException("Could not mutate layer from " + layerConfOpt)))
                .build())
        )).evolve();
        evaluateExperiment(decreaseKs, experimentLabel);
    }

    private void evalIncreaseKernelSize(ComputationGraph graph, String layer) {
        final String experimentLabel = "increaseKernelSize: ";
        final ModelAdapter increaseKs = new EvolvingGraphAdapter(graph, new MutateLayerContained(() -> Stream.of(MutateLayerContained.LayerMutation.builder()
                .layerName(layer)
                .inputLayers(getInputLayers(graph, layer))
                .mutation(layerConfOpt -> layerConfOpt
                        .map(Layer::clone)
                        .filter(layerConf -> layerConf instanceof ConvolutionLayer)
                        .map(layerConf -> (ConvolutionLayer) layerConf)
                        .map(convConf -> {
                            convConf.setKernelSize(new int[]{convConf.getKernelSize()[0] + 1, convConf.getKernelSize()[1]});
                            convConf.setWeightInit(WeightInit.DISTRIBUTION);
                            convConf.setDist(new ConstantDistribution(0));
                            return convConf;
                        })
                        .orElseThrow(() -> new IllegalArgumentException("Could not mutate layer from " + layerConfOpt)))
                .build())
        )).evolve();
        evaluateExperiment(increaseKs, experimentLabel);
    }

    private void evalDecreaseNoutOptimal(ComputationGraph graph, String layer) {
        final String experimentLabel = "decreaseNoutOpt: ";
        final Comparator<Integer> actContribComparator = getActivationContributionComparator(trainIter, layer, graph);
        final ModelAdapter decreaseNoutOpt = new EvolvingGraphAdapter(graph, new MutateNout(() -> Stream.of(MutateNout.NoutMutation.builder()
                .layerName(layer)
                .mutateNout(nOut -> nOut - 1)
                .build())
        ),
                graphVar -> new ParameterTransfer(graphVar,
                        layerName -> Optional.of(i -> actContribComparator))).evolve();
        evaluateExperiment(decreaseNoutOpt, experimentLabel);
    }

    private void evalDecreaseNout(ComputationGraph graph, String layer) {
        final String experimentLabel = "decreaseNout: ";
        final ModelAdapter decreaseNout = new EvolvingGraphAdapter(graph, new MutateNout(() -> Stream.of(MutateNout.NoutMutation.builder()
                .layerName(layer)
                .mutateNout(nOut -> nOut - 1)
                .build())
        )).evolve();
        evaluateExperiment(decreaseNout, experimentLabel);
    }

    private void evalIncreaseNout(ComputationGraph graph, String layer) {
        final String experimentLabel = "increaseNout: ";
        final ModelAdapter increaseNout = new EvolvingGraphAdapter(graph, new MutateNout(() -> Stream.of(MutateNout.NoutMutation.builder()
                .layerName(layer)
                .mutateNout(nOut -> nOut + 1)
                .build())
        )).evolve();
        evaluateExperiment(increaseNout, experimentLabel);
    }

    private void evalBaseline(ComputationGraph graph) {
        final ModelAdapter baseline = new GraphModelAdapter(graph);
        final String experimentLabel = "baseline: ";
        evaluateExperiment(baseline, experimentLabel);
    }

    private void evaluateExperiment(ModelAdapter baseline, String experimentLabel) {
        Evaluation evaluation = new Evaluation(evalIter.getLabels());
        baseline.eval(evalIter, evaluation);
        log.info(experimentLabel + evaluation.accuracy());
        evalIter.restartMiniEpoch();
    }

    private static String[] getInputLayers(ComputationGraph graph, String layerName) {
        return graph.getConfiguration().getVertexInputs().get(layerName).toArray(new String[] {});
    }

    private static Comparator<Integer> getActivationContributionComparator(
            MiniEpochDataSetIterator trainIter,
            String layer, ComputationGraph best) {

        final ComputationGraphConfiguration.GraphBuilder addEpsSpy = new ComputationGraphConfiguration.GraphBuilder(
                best.getConfiguration(),
                new NeuralNetConfiguration.Builder(best.conf())
        );
        final String epsSpyName = "EpsSpy_" + layer;
        MutateLayerContained.makeRoomFor(
                MutateLayerContained.LayerMutation.builder()
                        .layerName(epsSpyName)
                        .inputLayers(new String[]{layer})
                        .mutation(optLayer -> {
                            throw new IllegalArgumentException("Not expected! Vertex will be added afterwards!");
                        })
                        .build(),
                addEpsSpy
        );
        addEpsSpy.addVertex(epsSpyName, new EpsilonSpyVertex(), layer);

        class INDArrayCompsumer implements Comparator<Integer>, Consumer<INDArray> {

            private INDArray array;

            @Override
            public int compare(Integer o1, Integer o2) {
                return -Double.compare(array.getDouble(o1), array.getDouble(o2));
            }

            @Override
            public void accept(INDArray indArray) {
                if (array == null) {
                    array = indArray.detach();
                } else {
                    array.addi(indArray);
                }
            }
        }
        final INDArrayCompsumer actContribComparator = new INDArrayCompsumer();


        final ComputationGraph withEpsSpy = new ComputationGraph(addEpsSpy.build());
        withEpsSpy.init();
        new ParameterTransfer(best).transferWeightsTo(withEpsSpy);
        withEpsSpy.addListeners(new ActivationContribution("" + layer, arr -> {
            actContribComparator.accept(arr);
            log.debug("Contributions: " + arr);
        }));


        // Do a few training iterations to train the NoutMutation
        for (int i = 0; i < 3; i++) {
            withEpsSpy.fit(trainIter);
            trainIter.reset();
        }
        return actContribComparator;
    }
}
