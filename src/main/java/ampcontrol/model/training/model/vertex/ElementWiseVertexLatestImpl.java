package ampcontrol.model.training.model.vertex;
/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Or;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

/** An ElementWiseVertex is used to combine the activations of two or more layer in an element-wise manner<br>
 * For example, the activations may be combined by addition, subtraction or multiplication or by selecting the maximum.
 * Addition, Average, Product and Max may use an arbitrary number of input arrays. Note that in the case of subtraction, only two inputs may be used.
 * In all cases, the shape of the input arrays must be identical.
 * TODO: Remove when upgrading dl4j to 9.2-xxx
 * @author Alex Black
 */
public class ElementWiseVertexLatestImpl extends BaseGraphVertex {

    public enum Op {
        Add, Subtract, Product, Average
    }

    private Op op;
    private int nInForwardPass;

    public ElementWiseVertexLatestImpl(ComputationGraph graph, String name, int vertexIndex, Op op) {
        this(graph, name, vertexIndex, null, null, op);
    }

    public ElementWiseVertexLatestImpl(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                                       VertexIndices[] outputVertices, Op op) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.op = op;
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
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");

        nInForwardPass = inputs.length;
        if (inputs.length == 1)
            return inputs[0];

        switch (op) {
            case Add:
                INDArray sum = workspaceMgr.dup(ArrayType.ACTIVATIONS, inputs[0]);
                for (int i = 1; i < inputs.length; i++) {
                    sum.addi(inputs[i]);
                }
                return sum;
            case Average:
                INDArray average = workspaceMgr.dup(ArrayType.ACTIVATIONS, inputs[0]);
                for (int i = 1; i < inputs.length; i++) {
                    average.addi(inputs[i]);
                }
                return average.divi(inputs.length);
            case Subtract:
                if (inputs.length != 2)
                    throw new IllegalArgumentException("ElementWise subtraction only supports 2 inputs");
                return inputs[0].sub(inputs[1]);
            case Product:
                INDArray product = workspaceMgr.dup(ArrayType.ACTIVATIONS, inputs[0]);
                for (int i = 1; i < inputs.length; i++) {
                    product.muli(inputs[i]);
                }
                return product;
            default:
                throw new UnsupportedOperationException("Unknown op: " + this.op);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set");

        if (nInForwardPass == 1)
            return new Pair<>(null, new INDArray[] {epsilon});

        switch (op) {
            case Add:
                //If x=sum_i a_i then dL/da_i = dL/dx * dx/da_i = dL/dx
                INDArray[] out = new INDArray[nInForwardPass];
                for (int i = 0; i < nInForwardPass; i++)
                    out[i] = epsilon.dup();
                return new Pair<>(null, out);
            case Average:
                INDArray[] outAverage = new INDArray[nInForwardPass];
                for (int i = 0; i < nInForwardPass; i++)
                    outAverage[i] = epsilon.div(nInForwardPass);
                return new Pair<>(null, outAverage);
            case Subtract:
                INDArray[] out2 = new INDArray[2];
                out2[0] = epsilon;
                out2[1] = epsilon.neg();
                return new Pair<>(null, out2);
            case Product:
                INDArray[] out_product = new INDArray[nInForwardPass];
                for (int i = 0; i < nInForwardPass; i++) {
                    out_product[i] = epsilon.dup();
                    for (int j = 0; j < nInForwardPass; ++j) {
                        if (i != j)
                            out_product[i].muli(inputs[j]);
                    }
                }
                return new Pair<>(null, out_product);
            default:
                throw new UnsupportedOperationException("Unknown op: " + this.op);
        }
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new IllegalArgumentException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                                                           int minibatchSize) {
        if (maskArrays == null) {
            return new Pair<>(null, currentMaskState);
        }

        //Most common case: all or none.
        //If there's only *some* mask arrays: assume the others (missing) are equivalent to all 1s
        //And for handling multiple masks: best strategy seems to be an OR operation
        //i.e., output is 1 if any of the input are 1s
        //Which means: if any masks are missing, output null (equivalent to no mask, or all steps present)
        //Otherwise do an element-wise OR operation

        for (INDArray arr : maskArrays) {
            if (arr == null) {
                return new Pair<>(null, currentMaskState);
            }
        }

        //At this point: all present. Do OR operation
        if (maskArrays.length == 1) {
            return new Pair<>(maskArrays[0], currentMaskState);
        } else {
            INDArray ret = maskArrays[0].dup(maskArrays[0].ordering());
            Nd4j.getExecutioner().exec(new Or(maskArrays[0], maskArrays[1], ret));
            for (int i = 2; i < maskArrays.length; i++) {
                Nd4j.getExecutioner().exec(new Or(maskArrays[i], ret, ret));
            }
            return new Pair<>(ret, currentMaskState);
        }
    }

    @Override
    public String toString() {
        return "ElementWiseVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",op=" + op
                + ")";
    }
}