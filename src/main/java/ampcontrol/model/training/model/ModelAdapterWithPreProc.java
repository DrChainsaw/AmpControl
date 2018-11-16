package ampcontrol.model.training.model;

import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * {@link ModelAdapter} which decorates a source adapter by adding pre-processing from a {@link DataSetPreProcessor}
 * before {@link #fit(DataSetIterator) fit} and {@link #eval(DataSetIterator, IEvaluation[]) eval}
 */
public class ModelAdapterWithPreProc implements ModelAdapter {

    private final DataSetPreProcessor preProc;
    private final ModelAdapter adapter;

    /**
     * Constructor
     * @param preProc pre-processing to add
     * @param adapter adapter to be decorated
     */
    public ModelAdapterWithPreProc(DataSetPreProcessor preProc, ModelAdapter adapter) {
        this.preProc = preProc;
        this.adapter = adapter;
    }

    @Override
    public void fit(DataSetIterator iter) {
        DataSetPreProcessor existing = iter.getPreProcessor();
        iter.setPreProcessor(preProc);
        adapter.fit(iter);
        iter.setPreProcessor(existing);

    }

    @Override
    public <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals)  {
        DataSetPreProcessor existing = iter.getPreProcessor();
        iter.setPreProcessor(preProc);
        T[] result = adapter.eval(iter, evals);
        iter.setPreProcessor(existing);
        return result;
    }

    @Override
    public Model asModel() {
        return adapter.asModel();
    }
}
