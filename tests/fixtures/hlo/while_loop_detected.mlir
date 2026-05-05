%1 = stablehlo.constant dense<0> : tensor<i32>
%2 = stablehlo.while (%arg0 = %1) : (tensor<i32>) -> tensor<i32>
