package ampcontrol.model.training.model;

import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Adapter interface for {@link Model Models} which can be fitted and evaluated. Necessary to avoid code duplication
 * as I could not find a common interface for these two things in dl4j.
 *
 * @author Christian Skärby
 */
public interface ModelAdapter {

    /**
     * @see org.deeplearning4j.nn.api.Classifier#fit(DataSetIterator)
     * @see org.deeplearning4j.nn.graph.ComputationGraph#fit(DataSetIterator)
     */
    void fit(DataSetIterator iter);

    /**
     * @see org.deeplearning4j.nn.multilayer.MultiLayerNetwork#doEvaluation(DataSetIterator, IEvaluation[])
     * @see org.deeplearning4j.nn.graph.ComputationGraph#doEvaluation(DataSetIterator, IEvaluation[])
     */
    <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals);

    /**
     * Return the model as a {@link Model}
     * @return a {@link Model}
     */
    Model asModel();

}
