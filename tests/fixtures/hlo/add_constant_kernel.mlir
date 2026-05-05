%0 = stablehlo.constant dense<1.0> : tensor<f32>
%1 = stablehlo.constant dense<2.0> : tensor<f32>
%2 = stablehlo.add %0, %1 : tensor<f32>, tensor<f32> -> tensor<f32>
