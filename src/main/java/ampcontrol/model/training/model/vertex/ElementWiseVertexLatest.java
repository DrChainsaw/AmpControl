package ampcontrol.model.training.model.vertex;/*-
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

import lombok.Data;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * An ElementWiseVertex is used to combine the activations of two or more layer in an element-wise manner<br>
 * For example, the activations may be combined by addition, subtraction or multiplication or by selecting the maximum.
 * Addition, Average, Max and Product may use an arbitrary number of input arrays. Note that in the case of subtraction, only two inputs may be used.
 * TODO: Remove when upgrading dl4j to 9.2-xxx
 * @author Alex Black
 */
@Data
public class ElementWiseVertexLatest extends GraphVertex {

    protected Op op;

    public enum Op {
        Add, Subtract, Product, Average, Max
    }

    public ElementWiseVertexLatest(@JsonProperty("op") Op op) {
        this.op = op;
    }


    @Override
    public ElementWiseVertexLatest clone() {
        return new ElementWiseVertexLatest(op);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ElementWiseVertexLatest))
            return false;
        return ((ElementWiseVertexLatest) o).op == op;
    }

    @Override
    public int hashCode() {
        return op.hashCode();
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minVertexInputs() {
        return 2;
    }

    @Override
    public int maxVertexInputs() {
        switch (op) {
            case Add:
            case Average:
            case Product:
            case Max:
                //No upper bound
                return Integer.MAX_VALUE;
            case Subtract:
                return 2;
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams) {
        ElementWiseVertexLatestImpl.Op op;
        switch (this.op) {
            case Add:
                op = ElementWiseVertexLatestImpl.Op.Add;
                break;
            case Average:
                op = ElementWiseVertexLatestImpl.Op.Average;
                break;
            case Subtract:
                op = ElementWiseVertexLatestImpl.Op.Subtract;
                break;
            case Product:
                op = ElementWiseVertexLatestImpl.Op.Product;
                break;
            default:
                throw new UnsupportedOperationException("No support for op: " + this.op);
        }
        return new ElementWiseVertexLatestImpl(graph, name, idx, op);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length == 1)
            return vertexInputs[0];
        InputType first = vertexInputs[0];
        if (first.getType() != InputType.Type.CNN) {
            //FF, RNN or flat CNN data inputs
            for (int i = 1; i < vertexInputs.length; i++) {
                if (vertexInputs[i].getType() != first.getType()) {
                    throw new InvalidInputTypeException(
                            "Invalid input: ElementWise vertex cannot process activations of different types:"
                                    + " first type = " + first.getType() + ", input type " + (i + 1)
                                    + " = " + vertexInputs[i].getType());
                }
            }
        } else {
            //CNN inputs... also check that the depth, width and heights match:
            InputType.InputTypeConvolutional firstConv = (InputType.InputTypeConvolutional) first;
            int fd = firstConv.getDepth();
            int fw = firstConv.getWidth();
            int fh = firstConv.getHeight();

            for (int i = 1; i < vertexInputs.length; i++) {
                if (vertexInputs[i].getType() != InputType.Type.CNN) {
                    throw new InvalidInputTypeException(
                            "Invalid input: ElementWise vertex cannot process activations of different types:"
                                    + " first type = " + InputType.Type.CNN + ", input type " + (i + 1)
                                    + " = " + vertexInputs[i].getType());
                }

                InputType.InputTypeConvolutional otherConv = (InputType.InputTypeConvolutional) vertexInputs[i];

                int od = otherConv.getDepth();
                int ow = otherConv.getWidth();
                int oh = otherConv.getHeight();

                if (fd != od || fw != ow || fh != oh) {
                    throw new InvalidInputTypeException(
                            "Invalid input: ElementWise vertex cannot process CNN activations of different sizes:"
                                    + "first [depth,width,height] = [" + fd + "," + fw + "," + fh
                                    + "], input " + i + " = [" + od + "," + ow + "," + oh + "]");
                }
            }
        }
        return first; //Same output shape/size as
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory in addition to output activations
        return new LayerMemoryReport.Builder(null, ElementWiseVertexLatest.class, inputTypes[0], inputTypes[0])
                .standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0).build();
    }
}