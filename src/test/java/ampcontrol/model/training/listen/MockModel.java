package ampcontrol.model.training.listen;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;
import java.util.Map;

public class MockModel implements Model {

    @Override
    public void init() {
        //Ignore
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        //Ignore
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        //Ignore
    }

    @Override
    public void addListeners(IterationListener... listener) {
        //Ignore
    }

    @Override
    public void fit() {
        //Ignore
    }

    @Override
    public void update(Gradient gradient) {
        //Ignore
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        //Ignore
    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void computeGradientAndScore() {
        //Ignore
    }

    @Override
    public void accumulateScore(double accum) {
        //Ignore
    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public int numParams(boolean backwards) {
        return 0;
    }

    @Override
    public void setParams(INDArray params) {
        //Ignore
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        //Ignore
    }

    @Override
    public INDArray getGradientsViewArray() {
        return null;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        //Ignore
    }

    @Override
    public void applyLearningRateScoreDecay() {
        //Ignore
    }

    @Override
    public void fit(INDArray data) {
        //Ignore
    }

    @Override
    public void iterate(INDArray input) {
        //Ignore
    }

    @Override
    public Gradient gradient() {
        return null;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return null;
    }

    @Override
    public int batchSize() {
        return 0;
    }

    @Override
    public NeuralNetConfiguration conf() {
        return null;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        //Ignore
    }

    @Override
    public INDArray input() {
        return null;
    }

    @Override
    public void validateInput() {
        //Ignore
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return null;
    }

    @Override
    public void initParams() {
        //Ignore
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return null;
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        return null;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        //Ignore
    }

    @Override
    public void setParam(String key, INDArray val) {
        //Ignore
    }

    @Override
    public void clear() {
        //Ignore
    }
}
