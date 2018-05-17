package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link Conv1DLstmDenseFactory}
 *
 * @author Christian Skärby
 */
public class Conv1DLstmDenseFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer().initialize(
                ((tr, ev, is, np, md) -> new Conv1DLstmDenseFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
        );
    }


}