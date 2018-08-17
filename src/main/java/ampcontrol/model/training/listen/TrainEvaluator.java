package ampcontrol.model.training.listen;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;

/**
 * {@link BaseTrainingListener} which evaluates the given {@link Model} and notifies a {@link BiConsumer} of the training
 * accuracy per iteration.
 * TODO: Too painful to test due to dependencies to Dl4j internals. Possible to redesign?
 *
 * @author Christian Skärby
 */
public class TrainEvaluator extends BaseTrainingListener {

    private static final Logger log = LoggerFactory.getLogger(TrainEvaluator.class);

    private static final LayerWorkspaceMgr wsMgr = LayerWorkspaceMgr.noWorkspaces();

    private final BiConsumer<Integer, Double> iterAndEvalListener;

    private final Evaluation eval;

    private int iterStore = 0;
    private int epochStore = 0;
    private INDArray labels;
    private INDArray output;


    public TrainEvaluator(
            BiConsumer<Integer, Double> iterAnEvalListener) {
        this.iterAndEvalListener = iterAnEvalListener;
        this.eval = new Evaluation();
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        eval.eval(labels, output);
        iterStore = iteration;
        epochStore = epoch;

    }

    @Override
    public void onEpochStart(Model model) {
        eval.reset();
    }

    @Override
    public void onEpochEnd(Model model) {
        if (model instanceof ComputationGraph) {
            final NeuralNetConfiguration conf = ((ComputationGraph) model).getOutputLayer(0).conf();
            log.info("Training accuracy at iteration " + iterStore + ": " + eval.accuracy() + " curr learning rate: " + conf.getLayer().getUpdaterByParam("").getLearningRate(iterStore, epochStore));
            iterAndEvalListener.accept(iterStore, eval.accuracy());
        }
    }

    private INDArray getActivation(Map<String, INDArray> activations, GraphVertex[] vertices, int vertexInd) {
        final GraphVertex vertex = vertices[vertexInd];
        return Optional.ofNullable(activations.get(vertex.getVertexName())).orElseGet(() -> {
            final boolean inputsNull = vertex.getInputs() == null;
            if (inputsNull) {
                final VertexIndices[] inputInds = vertex.getInputVertices();
                for (VertexIndices inputInd : inputInds) {
                    vertex.setInput(inputInd.getVertexEdgeNumber(), getActivation(activations, vertices, inputInd.getVertexIndex()), wsMgr);
                }
            }
            final INDArray activation = vertex.doForward(false, wsMgr).detach();
            if (inputsNull) {
                vertex.clear();
            }
            return activation;
        });
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        if (model instanceof ComputationGraph) {
            final ComputationGraph graph = (ComputationGraph) model;
            final Layer ol = graph.getOutputLayer(0);
            output = getActivation(activations, graph.getVertices(), ol.getIndex());
        } else {
            throw new UnsupportedOperationException("Not implemented!");
        }
    }

    @Override
    public void onBackwardPass(Model model) {
        if (model instanceof ComputationGraph) {
            final BaseOutputLayer ol = (BaseOutputLayer) ((ComputationGraph) model).getOutputLayer(0);
            this.labels = ol.getLabels();
        } else {
            throw new UnsupportedOperationException("Not implemented!");
        }
    }
}
