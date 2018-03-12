/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package ampControl.model.training.model.vertex;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * Ripped out of dl4j master and converted to a vertex as I couldn't figure out how to do custom layers.
 * TODO: Remove when upgrading dl4j to 9.2-xxx
 */
public class ZeroPadding1DVertexImpl extends BaseGraphVertex {

    private final int[] padding; // [padLeft, padRight]

    public ZeroPadding1DVertexImpl(ComputationGraph graph, String name, int vertexIndex, int[] padding) {
        this(graph, name, vertexIndex, null, null, padding);

    }

    public ZeroPadding1DVertexImpl(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices, int[] padding) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.padding = padding;
    }

    @Override
    public String toString() {
        return "zeroPadding1DVertex" + Arrays.toString(padding);
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training) {

        int[] inShape = getInputs()[0].shape();
        int paddedOut = inShape[2] + padding[0] + padding[1];
        int[] outShape = new int[] {inShape[0], inShape[1], paddedOut};

        INDArray out = Nd4j.create(outShape);
        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2])}, getInputs()[0]);

        return out;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        int[] inShape = inputs[0].shape();

        INDArray epsNext = epsilon.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2]));

        return new Pair<>(new DefaultGradient(), new INDArray[] {epsNext});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {

    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}