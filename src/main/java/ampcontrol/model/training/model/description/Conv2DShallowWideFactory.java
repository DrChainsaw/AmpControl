package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;

/**
 * Description of a bunch of architectures which belong to a family of shallow but wide 2D convolutional neural networks.
 * Idea comes from some arxiv paper, but I can't remember which one. Seems to perform worse than 2D CNNs with small
 * kernel sizes.
 *
 * @author Christian Skärby
 */
public class Conv2DShallowWideFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public Conv2DShallowWideFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelDir = modelDir;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {

        DoubleStream.of(0).forEach(dropOutProb -> {
            int kernelSizeLong = inputShape[1] - 2; // Allow for 2 frequency bins invariance
            // int kernelSizeHalf = inputShape[1] / 2;
            int poolSizeTime = inputShape[0] / 2;
            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                    new BlockBuilder()
                            .setStartingLearningRate(0.0005)
                            .setUpdater(new Nesterovs(0.9))
                            .setNamePrefix(namePrefix)
                            .first(new ConvType(inputShape))
                            //.multiLevel()
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize_w(kernelSizeLong)
                                    .setKernelSize_h(2)
                                    .setNrofKernels(128))
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setKernelSize_w(kernelSizeHalf)
//                            .setKernelSize_h(2)
//                            .setStride_w(kernelSizeHalf / 8)
//                            .setNrofKernels(128))
                            //  .done()
                            //.andThen(new DropOut().setDropProb(dropOutProb))
                            .andThen(new Pool2D().setSize_h(poolSizeTime).setSize_w(1).setStride_h(poolSizeTime / 8).setSize_w(1))
                            .andThen(new Conv2DBatchNormAfter()
                                    .setKernelSize_w(3)
                                    .setKernelSize_h(3)
                                    .setNrofKernels(128))
                            //.andThen(new GlobMeanMax())
                            .andThen(new Dense()
                                    .setHiddenWidth(256)
                                    .setActivation(new ActivationReLU()))
                            .andThen(new DropOut().setDropProb(dropOutProb))
                            .andFinally(new Output(trainIter.totalOutcomes())));
            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new GraphModelAdapter(builder.buildGraph()),
                    builder.name()));
        });
    }
}
