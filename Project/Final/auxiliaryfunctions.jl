using DataFrames
using Distances, LinearAlgebra
using Images, ImageContrastAdjustment, ImageCore, ImageTransformations, Rotations, CoordinateTransformations

function applyrotate(image, rotation=pi/2)
    trfm = recenter(RotMatrix(rotation), center(image));
    mimg = warp(image,trfm)
    mimg = mimg[1:size(image)[1],1:size(image)[2]]
    return mimg
	# return mimg
end

function applygammatransform(image, gamma = 2)
    mimg = adjust_histogram(image, GammaCorrection(gamma = gamma))
    return mimg
end

function applytransformations(image, gamma = 0.1, rotation = 0, mirrorv=false, mirrorh=false)
    mimg = applygammatransform(image,gamma)
	mimg = applyrotate(mimg,rotation)
	if mirrorv == true
		mimg = reverse(mimg,dims=1)
	end
	if mirrorh == true
		mimg = reverse(mimg,dims=2)
	end
    return mimg
end
