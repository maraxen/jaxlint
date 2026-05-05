%0 = stablehlo.constant dense<1.0> : tensor<4xf32>
%1 = stablehlo.reshape %0 : (tensor<4xf32>) -> tensor<2x2xf32>
%2 = stablehlo.constant dense<2.0> : tensor<2x2xf32>
%3 = stablehlo.multiply %1, %2 : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
