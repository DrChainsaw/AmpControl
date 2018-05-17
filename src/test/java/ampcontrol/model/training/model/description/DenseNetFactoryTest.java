package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link DenseNetFactory}
 *
 * @author Christian Skärby
 */
public class DenseNetFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer().initialize(
                ((tr, ev, is, np, md) -> new DenseNetFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
        );
    }
}